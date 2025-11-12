#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
DATA_PATH = Path("../data/Completed Australian case set.xlsx")
OUTPUTS   = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
PNG_DEN   = OUTPUTS / "AUS climate case network density over time.png"

# ---------- helper functions ----------
def clean_id(val):
    '''Canonical string ID: trims; 12.0 -> '12'; drops empties/NaN.'''
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
        return s

def parse_ids_from_cell(cell):
    '''Extract all integer IDs from a free-text cell (e.g., '1, 2; 3\n[4]').'''
    if pd.isna(cell):
        return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

# ---------- load and clean ----------
df = pd.read_excel(DATA_PATH)
ID_COL, YEAR_COL, CITED_BY = df.columns[0], df.columns[3], df.columns[4]

df[ID_COL] = df[ID_COL].apply(clean_id)
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df = df[df[ID_COL].notna()].copy()
id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

# ---------- compute yearly density ----------
rows = []
for year in range(1994, 2025):
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
    valid_ids = set(sub[ID_COL].dropna().tolist())
    n = len(valid_ids)

    if n < 2:
        rows.append({"Year": year, "Density": 0.0})
        continue

    m = 0
    for _, row in sub.iterrows():
        target = row[ID_COL]
        for citer in parse_ids_from_cell(row.get(CITED_BY)):
            if citer and citer != target and citer in valid_ids:
                citer_year = id_to_year.get(citer, np.nan)
                if not pd.isna(citer_year) and citer_year <= year:
                    m += 1

    possible_edges = n * (n - 1)
    density = (m / possible_edges) if possible_edges else 0.0
    rows.append({"Year": year, "Density": density})

summary_df = pd.DataFrame(rows)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(summary_df["Year"], summary_df["Density"], marker='o', linewidth=2.5, markersize=5, color='tab:purple')
ax.set_title("Australian Climate Case Network Density Over Time", fontsize=14, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Network Density", fontsize=12)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PNG_DEN, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"Saved figure -> {PNG_DEN.resolve()}")
