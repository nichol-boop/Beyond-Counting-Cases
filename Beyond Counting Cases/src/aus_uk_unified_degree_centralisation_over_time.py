#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
from pathlib import Path

# ---------- paths ----------
AUS_PATH = Path("../data/Completed Australian case set.xlsx")
UK_PATH  = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS  = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
CSV_OUT  = OUTPUTS / "AUS_UK_unified_degree_centralisation_over_time.csv"
PNG_OUT  = OUTPUTS / "AUS_UK_unified_degree_centralisation_over_time.png"

# ---------- helpers ----------
def clean_id(val):
    """Normalise case IDs so citation targets match sources."""
    if pd.isna(val): 
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan": 
        return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        if s.endswith(".0"): 
            s = s[:-2]
        return s.replace("\u200b", "").strip()

def freeman_centralisation(degrees: np.ndarray, directed: bool = True) -> float:
    """Freeman degree centralisation formula."""
    n = len(degrees)
    if n < 2:
        return 0.0
    dmax = degrees.max() if n else 0.0
    sum_diff = float(np.sum(dmax - degrees))
    denom = (n - 1) ** 2 if directed else (n - 1) * (n - 2)
    return (sum_diff / denom) if denom > 0 else 0.0

def parse_ids_from_cell(cell):
    """Extract all integer IDs from a free-text citation field."""
    if pd.isna(cell):
        return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

# ---------- computation functions ----------
def compute_aus_centralisation(path):
    df = pd.read_excel(path)
    ID_COL = df.columns[0]      # A: Case ID
    YEAR_COL = df.columns[3]    # D: Year
    CITED_BY = df.columns[4]    # E: Cited by
    df[ID_COL] = df[ID_COL].apply(clean_id)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[ID_COL].notna()].copy()
    id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

    def build_edges_for_year(valid_ids, year_cutoff):
        edges = []
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year_cutoff)]
        for _, row in sub.iterrows():
            tgt = row[ID_COL]
            for src in parse_ids_from_cell(row[CITED_BY]):
                if not src or src == tgt or src not in valid_ids:
                    continue
                src_year = id_to_year.get(src, np.nan)
                if pd.isna(src_year) or src_year > year_cutoff:
                    continue
                edges.append((src, tgt))
        return edges

    rows = []
    for year in range(1994, 2025):
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)]
        ids = sub[ID_COL].dropna().tolist()
        valid_ids = set(ids)
        n = len(valid_ids)
        if n < 2:
            rows.append({"Year": year, "UnifiedDegreeCentralisation": 0.0})
            continue
        edges = build_edges_for_year(valid_ids, year)
        indeg = {i: 0 for i in valid_ids}
        outdeg = {i: 0 for i in valid_ids}
        for u, v in edges:
            outdeg[u] += 1
            indeg[v] += 1
        totaldeg_arr = np.array([indeg[i] + outdeg[i] for i in ids], dtype=float)
        unified_cent = freeman_centralisation(totaldeg_arr, directed=True)
        rows.append({"Year": year, "UnifiedDegreeCentralisation": unified_cent})
    return pd.DataFrame(rows)

def compute_uk_centralisation(path):
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

    def build_edges_for_year(valid_ids, year_cutoff):
        edges = []
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year_cutoff)]
        for _, row in sub.iterrows():
            src = row[ID_COL]
            for col in CITATION_COLS:
                tgt = row[col]
                if not tgt or tgt not in valid_ids or tgt == src:
                    continue
                tgt_year = id_to_year.get(tgt, np.nan)
                if pd.isna(tgt_year) or tgt_year > year_cutoff:
                    continue
                edges.append((src, tgt))
        return edges

    rows = []
    for year in range(1995, 2025):
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)]
        ids = sub[ID_COL].dropna().tolist()
        valid_ids = set(ids)
        n = len(valid_ids)
        if n < 2:
            rows.append({"Year": year, "UnifiedDegreeCentralisation": 0.0})
            continue
        edges = build_edges_for_year(valid_ids, year)
        indeg = {i: 0 for i in valid_ids}
        outdeg = {i: 0 for i in valid_ids}
        for u, v in edges:
            outdeg[u] += 1
            indeg[v] += 1
        totaldeg_arr = np.array([indeg[i] + outdeg[i] for i in ids], dtype=float)
        unified_cent = freeman_centralisation(totaldeg_arr, directed=True)
        rows.append({"Year": year, "UnifiedDegreeCentralisation": unified_cent})
    return pd.DataFrame(rows)

# ---------- compute ----------
aus_df = compute_aus_centralisation(AUS_PATH)
uk_df = compute_uk_centralisation(UK_PATH)

# Align years
merged_df = pd.merge(aus_df, uk_df, on="Year", how="outer", suffixes=("_AUS", "_UK")).sort_values("Year")
merged_df.to_csv(CSV_OUT, index=False, encoding="utf-8")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(merged_df["Year"], merged_df["UnifiedDegreeCentralisation_AUS"],
        marker='o', linewidth=2.2, color='tab:blue', label='Australia', alpha=0.8)
ax.plot(merged_df["Year"], merged_df["UnifiedDegreeCentralisation_UK"],
        marker='s', linewidth=2.2, color='tab:orange', label='United Kingdom', alpha=0.8)

ax.set_title("Unified Degree Centralisation Over Time: Australia vs United Kingdom (1994–2024)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Centralisation (0–1)", fontsize=11)
ax.set_ylim(0, 0.25)   # <-- Zoomed in to 0.25
ax.grid(alpha=0.3)
ax.legend(fontsize=10, frameon=True)

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"✅ Saved CSV -> {CSV_OUT.resolve()}")
print(f"✅ Saved Figure -> {PNG_OUT.resolve()}")
