#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

# ---------- paths ----------
DATA_PATH = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS   = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
CSV_OUT   = OUTPUTS / "UK unified degree centralisation over time.csv"
PNG_CENT  = OUTPUTS / "UK unified degree centralisation over time.png"

# ---------- load ----------
df = pd.read_excel(DATA_PATH)

# Identify columns
YEAR_COL = "Year decided" if "Year decided" in df.columns else df.columns[3]
CAND_ID_COLS = ["Node ID", "ID", "Case ID", "node_id", "Id"]
ID_COL = next((c for c in CAND_ID_COLS if c in df.columns), df.columns[1])
CITATION_COLS = df.columns[4:].tolist()

# ---------- helpers ----------
def clean_id(val):
    """Normalise case IDs so citation targets match sources."""
    if pd.isna(val): return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan": return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        if s.endswith(".0"): s = s[:-2]
        return s.replace("\u200b", "").strip()

df = df.copy()
df[ID_COL] = df[ID_COL].apply(clean_id)
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
for col in CITATION_COLS:
    df[col] = df[col].apply(clean_id)

# Map of ID → year
id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

def freeman_centralisation(degrees: np.ndarray, directed: bool = True) -> float:
    """Freeman degree centralisation formula."""
    n = len(degrees)
    if n < 2:
        return 0.0
    dmax = degrees.max() if n else 0.0
    sum_diff = float(np.sum(dmax - degrees))
    denom = (n - 1) ** 2 if directed else (n - 1) * (n - 2)
    return (sum_diff / denom) if denom > 0 else 0.0

def build_edges_for_year(valid_ids: set, year_cutoff: int, allow_same_year=True):
    """Build directed edges (src -> tgt) among cases with YEAR <= year_cutoff."""
    edges = []
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year_cutoff) & (df[ID_COL].isin(valid_ids))]
    for _, row in sub.iterrows():
        src = row[ID_COL]; src_year = row[YEAR_COL]
        for col in CITATION_COLS:
            tgt = row[col]
            if not tgt or tgt not in valid_ids or tgt == src:
                continue
            tgt_year = id_to_year.get(tgt, np.nan)
            if pd.isna(tgt_year): continue
            # enforce temporal logic (no future citations)
            if allow_same_year:
                if tgt_year > year_cutoff: continue
            else:
                if tgt_year >= src_year: continue
            edges.append((src, tgt))
    return edges

# ---------- compute unified centralisation over time ----------
rows = []
for year in range(1995, 2025):
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
    ids = sub[ID_COL].dropna().tolist()
    valid_ids = set(ids)
    n = len(valid_ids)

    if n < 2:
        rows.append({"Year": year, "Nodes": n, "Edges": 0,
                     "UnifiedCentralisation": 0.0})
        continue

    edges = build_edges_for_year(valid_ids, year_cutoff=year, allow_same_year=True)

    indeg = {i: 0 for i in valid_ids}
    outdeg = {i: 0 for i in valid_ids}
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1

    totaldeg_arr = np.array([indeg[i] + outdeg[i] for i in ids], dtype=float)
    unified_cent = freeman_centralisation(totaldeg_arr, directed=True)

    rows.append({
        "Year": year,
        "Nodes": n,
        "Edges": len(edges),
        "UnifiedDegreeCentralisation": unified_cent
    })

summary_df = pd.DataFrame(rows)

# ---------- save ----------
summary_df.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"Saved CSV -> {CSV_OUT.resolve()}\n")

# ---------- console summary ----------
print("Unified Degree Centralisation (last 10 years):")
print("=" * 70)
print(summary_df.tail(10).to_string(index=False, float_format="%.4f"))

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(summary_df["Year"], summary_df["UnifiedDegreeCentralisation"],
        marker='o', linewidth=2.2, color='tab:purple', alpha=0.8)
ax.set_title("UK Climate Case Network: Unified Degree Centralisation Over Time (1995–2024)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Centralisation (0–1)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PNG_CENT, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"Saved figure -> {PNG_CENT.resolve()}")
