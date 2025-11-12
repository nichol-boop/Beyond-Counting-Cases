#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ---------- paths ----------
DATA_PATH = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS   = Path("../outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)
CSV_OUT   = OUTPUTS / "UK degree centralisation over time.csv"
PNG_CENT  = OUTPUTS / "UK degree centralisation over time.png"

# ---------- load ----------
df = pd.read_excel(DATA_PATH)

# Identify columns
YEAR_COL = "Year decided" if "Year decided" in df.columns else df.columns[3]
CAND_ID_COLS = ["Node ID", "ID", "Case ID", "node_id", "Id"]
ID_COL = next((c for c in CAND_ID_COLS if c in df.columns), df.columns[1])

CITATION_COLS = df.columns[4:].tolist()  # E onward, as in your density code

# ---- helpers ----
def clean_id(val):
    """Normalise IDs so targets match sources (strip, drop .0, handle floats/ints/strings)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    # If looks numeric, convert to int when possible (so 12.0 -> "12")
    try:
        f = float(s.replace(",", ""))  # remove commas if any
        if math.isfinite(f) and float(f).is_integer():
            return str(int(f))
        # else leave as precise float string
        return str(f)
    except Exception:
        # non-numeric IDs (allow alphanumeric IDs if your dataset uses them)
        # Remove zero-width spaces and normalise whitespace
        s = s.replace("\u200b", "").strip()
        # Drop trailing ".0" if present from Excel export
        if s.endswith(".0"):
            s = s[:-2]
        return s

# Clean core columns
df = df.copy()
df[ID_COL] = df[ID_COL].apply(clean_id)
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")

# Pre-clean a copy of all citation cells to avoid repeated string parsing in the loop
for col in CITATION_COLS:
    df[col] = df[col].apply(clean_id)

# Map of ID -> year (for ensuring targets exist & are not from the future)
id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

def freeman_degree_centralisation(degrees: np.ndarray, directed: bool = True) -> float:
    """
    Freeman degree centralisation. For directed degree centralisation (in/out),
    the star-graph normaliser is (n-1)^2.
    """
    n = len(degrees)
    if n < 2:
        return 0.0
    dmax = degrees.max() if n else 0.0
    sum_diff = float(np.sum(dmax - degrees))
    denom = (n - 1) ** 2 if directed else (n - 1) * (n - 2)
    return (sum_diff / denom) if denom > 0 else 0.0

def build_edges_for_year(valid_ids: set, year_cutoff: int, allow_same_year: bool = True):
    """
    Build directed edges (src -> tgt) among cases with YEAR <= year_cutoff.
    Only keep edges where both src and tgt are in valid_ids and tgt's year <= cutoff.
    """
    edges = []
    # Filter rows up-front
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year_cutoff) & (df[ID_COL].isin(valid_ids))]

    for _, row in sub.iterrows():
        src = row[ID_COL]
        src_year = row[YEAR_COL]
        for col in CITATION_COLS:
            tgt = row[col]
            if not tgt:
                continue
            if tgt not in valid_ids:
                continue
            tgt_year = id_to_year.get(tgt, np.nan)
            if pd.isna(tgt_year):
                continue
            # Enforce temporal logic: citations should not go to the future
            if allow_same_year:
                if tgt_year > year_cutoff:
                    continue
            else:
                if tgt_year >= src_year:  # strict earlier than the citing case
                    continue
            if tgt != src:
                edges.append((src, tgt))
    return edges

# ---------- compute centralisation over time ----------
rows = []
for year in range(1995, 2025):
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
    ids = sub[ID_COL].dropna().tolist()
    valid_ids = set(ids)
    n = len(valid_ids)

    if n < 2:
        rows.append({"Year": year, "Nodes": n, "Edges": 0,
                     "InDegreeCentralisation": 0.0, "OutDegreeCentralisation": 0.0})
        continue

    edges = build_edges_for_year(valid_ids, year_cutoff=year, allow_same_year=True)

    # degree dictionaries
    indeg = {i: 0 for i in valid_ids}
    outdeg = {i: 0 for i in valid_ids}
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1

    indeg_arr = np.array([indeg[i] for i in ids], dtype=float)
    outdeg_arr = np.array([outdeg[i] for i in ids], dtype=float)

    in_cent = freeman_degree_centralisation(indeg_arr, directed=True)
    out_cent = freeman_degree_centralisation(outdeg_arr, directed=True)

    rows.append({
        "Year": year,
        "Nodes": n,
        "Edges": len(edges),
        "InDegreeCentralisation": in_cent,
        "OutDegreeCentralisation": out_cent
    })

summary_df = pd.DataFrame(rows)

# ---------- save CSV ----------
summary_df.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"Saved CSV -> {CSV_OUT.resolve()}")

# ---------- sanity prints ----------
print("\nDegree Centralisation by Year (last 8 years):")
print("=" * 80)
print(summary_df.tail(8).to_string(index=False, float_format="%.4f"))

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(summary_df["Year"], summary_df["InDegreeCentralisation"],
        marker='o', linewidth=2.5, markersize=6, 
        label="In-degree centralisation (Citations Received)",
        color='tab:red', alpha=0.8)

ax.plot(summary_df["Year"], summary_df["OutDegreeCentralisation"],
        marker='s', linewidth=2.5, markersize=6, 
        label="Out-degree centralisation (Citations Made)",
        color='tab:blue', alpha=0.8)

ax.set_title("UK Climate Case Network: Degree Centralisation Over Time (1995–2024)", 
             fontsize=14, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Centralisation (0–1)", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xticks(list(range(1995, 2025, 5)))
ax.grid(True, alpha=0.3)
ax.legend(loc='best', framealpha=0.9, fontsize=10)

fig.tight_layout()
fig.savefig(PNG_CENT, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved figure -> {PNG_CENT.resolve()}")
