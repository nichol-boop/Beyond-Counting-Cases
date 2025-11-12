#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
AUS_PATH = Path("../data/Completed Australian case set.xlsx")
UK_PATH  = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS  = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
PNG_OUT  = OUTPUTS / "AUS_vs_UK_climate_case_network_density_over_time.png"

# ---------- helpers ----------
def clean_id(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan": return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s[:-2] if s.endswith(".0") else s

def parse_ids_from_cell(cell):
    """Extract all integer IDs from a text cell."""
    if pd.isna(cell): return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

# ---------- Australia ----------
def calculate_aus_density(path):
    df = pd.read_excel(path)
    ID_COL, YEAR_COL, CITED_BY = df.columns[0], df.columns[3], df.columns[4]
    df[ID_COL] = df[ID_COL].apply(clean_id)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[ID_COL].notna()].copy()
    id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

    out = []
    for year in range(1994, 2025):
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
        valid_ids = set(sub[ID_COL].dropna().tolist())
        n = len(valid_ids)
        if n < 2:
            out.append({"Year": year, "Density": 0.0})
            continue

        m = 0
        for _, row in sub.iterrows():
            target = row[ID_COL]
            for citer in parse_ids_from_cell(row.get(CITED_BY)):
                if not citer or citer == target or citer not in valid_ids:
                    continue
                citer_year = id_to_year.get(citer, np.nan)
                if not pd.isna(citer_year) and citer_year <= year:
                    m += 1

        possible_edges = n * (n - 1)
        density = (m / possible_edges) if possible_edges else 0.0
        out.append({"Year": year, "Density": density})
    return pd.DataFrame(out)

# ---------- United Kingdom ----------
def calculate_uk_density(path):
    df = pd.read_excel(path)
    out = []
    for year in range(1995, 2025):
        sub = df[pd.to_numeric(df['Year decided'], errors='coerce') <= year].copy()
        n = len(sub)
        if n < 2:
            out.append({"Year": year, "Density": 0.0})
            continue
        citation_cols = df.columns[4:].tolist()
        m = 0
        for _, row in sub.iterrows():
            for col in citation_cols:
                if pd.notna(row[col]) and str(row[col]).strip() != "":
                    m += 1
        possible = n * (n - 1)
        out.append({"Year": year, "Density": (m / possible) if possible else 0.0})
    return pd.DataFrame(out)

# ---------- compute ----------
aus_df = calculate_aus_density(AUS_PATH)
uk_df  = calculate_uk_density(UK_PATH)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(aus_df["Year"], aus_df["Density"], marker='o', linewidth=2.5, markersize=6,
        color='tab:purple', label='Australia')
ax.plot(uk_df["Year"], uk_df["Density"], marker='s', linewidth=2.5, markersize=6,
        color='tab:orange', label='United Kingdom')

ax.set_title("Climate Case Network Density Over Time: Australia vs UK", fontsize=18, fontweight='bold')
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Network Density", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(loc='upper left', frameon=True, fontsize=12)

ymax = max(aus_df["Density"].max(), uk_df["Density"].max())
ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 0.01)

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"Saved combined figure -> {PNG_OUT.resolve()}")
