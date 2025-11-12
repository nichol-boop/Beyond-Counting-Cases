#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math, re

# ---------- paths ----------
AUS_PATH = Path("../data/Completed Australian case set.xlsx")
UK_PATH  = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS  = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)

CSV_OUT  = OUTPUTS / "AUS_UK_in_degree_centralisation_over_time.csv"
PNG_OUT  = OUTPUTS / "AUS_UK_in_degree_centralisation_over_time.png"

# ---------- helpers ----------
def clean_id(val):
    """Normalise IDs so targets match sources (strip, drop .0, handle floats/ints/strings)."""
    if pd.isna(val): return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan": return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f) and f.is_integer():  # 12.0 -> "12"
            return str(int(f))
        return str(f)
    except Exception:
        s = s.replace("\u200b", "").strip()
        if s.endswith(".0"): s = s[:-2]
        return s

def parse_ids_from_cell(cell):
    """Extract integer case IDs from a free-text cell (e.g., '1, 2; [3]')."""
    if pd.isna(cell): return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

def freeman_in_degree_centralisation(in_degrees: np.ndarray) -> float:
    """
    Freeman degree centralisation for directed in-degrees.
    Normaliser for a directed star is (n-1)^2.
    """
    n = len(in_degrees)
    if n < 2: 
        return 0.0
    dmax = in_degrees.max()
    sum_diff = float(np.sum(dmax - in_degrees))
    denom = (n - 1) ** 2
    return (sum_diff / denom) if denom > 0 else 0.0

# ---------- Australia: in-degree centralisation over time ----------
def in_degree_centralisation_aus(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    ID_COL, YEAR_COL, CITED_BY = df.columns[0], df.columns[3], df.columns[4]

    df[ID_COL] = df[ID_COL].apply(clean_id)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[ID_COL].notna()].copy()

    id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

    rows = []
    for year in range(1994, 2025):
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)]
        ids = sub[ID_COL].dropna().tolist()
        valid_ids = set(ids)
        n = len(valid_ids)
        if n < 2:
            rows.append({"Year": year, "InDegreeCentralisation_AUS": 0.0})
            continue

        # Build edges citer -> target within cutoff
        indeg = {i: 0 for i in valid_ids}
        for _, row in sub.iterrows():
            tgt = row[ID_COL]
            for src in parse_ids_from_cell(row[CITED_BY]):
                if not src or src == tgt or src not in valid_ids:
                    continue
                src_y = id_to_year.get(src, np.nan)
                if pd.isna(src_y) or src_y > year:
                    continue
                indeg[tgt] += 1

        indeg_arr = np.array([indeg[i] for i in ids], dtype=float)
        in_cent = freeman_in_degree_centralisation(indeg_arr)
        rows.append({"Year": year, "InDegreeCentralisation_AUS": in_cent})

    return pd.DataFrame(rows)

# ---------- United Kingdom: in-degree centralisation over time ----------
def in_degree_centralisation_uk(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    YEAR_COL = "Year decided" if "Year decided" in df.columns else df.columns[3]
    CAND_ID_COLS = ["Node ID", "ID", "Case ID", "node_id", "Id"]
    ID_COL = next((c for c in CAND_ID_COLS if c in df.columns), df.columns[1])
    CITATION_COLS = df.columns[4:].tolist()

    df[ID_COL] = df[ID_COL].apply(clean_id)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    for col in CITATION_COLS:
        df[col] = df[col].apply(clean_id)

    id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

    rows = []
    for year in range(1995, 2025):
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)]
        ids = sub[ID_COL].dropna().tolist()
        valid_ids = set(ids)
        n = len(valid_ids)
        if n < 2:
            rows.append({"Year": year, "InDegreeCentralisation_UK": 0.0})
            continue

        indeg = {i: 0 for i in valid_ids}
        for _, row in sub.iterrows():
            src = row[ID_COL]
            for col in CITATION_COLS:
                tgt = row[col]
                if not tgt or tgt == src or tgt not in valid_ids:
                    continue
                tgt_y = id_to_year.get(tgt, np.nan)
                if pd.isna(tgt_y) or tgt_y > year:
                    continue
                indeg[tgt] += 1

        indeg_arr = np.array([indeg[i] for i in ids], dtype=float)
        in_cent = freeman_in_degree_centralisation(indeg_arr)
        rows.append({"Year": year, "InDegreeCentralisation_UK": in_cent})

    return pd.DataFrame(rows)

# ---------- compute ----------
aus_df = in_degree_centralisation_aus(AUS_PATH)
uk_df  = in_degree_centralisation_uk(UK_PATH)

merged = pd.merge(aus_df, uk_df, on="Year", how="outer").sort_values("Year")
merged.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"Saved CSV -> {CSV_OUT.resolve()}")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(merged["Year"], merged["InDegreeCentralisation_AUS"],
        marker='o', linewidth=2.5, markersize=6, label="Australia", color='tab:purple')
ax.plot(merged["Year"], merged["InDegreeCentralisation_UK"],
        marker='s', linewidth=2.5, markersize=6, label="United Kingdom", color='tab:orange')

ax.set_title("In-Degree Centralisation Over Time: Australia vs United Kingdom (1994–2024)",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("In-degree centralisation (0–1)", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"Saved figure -> {PNG_OUT.resolve()}")