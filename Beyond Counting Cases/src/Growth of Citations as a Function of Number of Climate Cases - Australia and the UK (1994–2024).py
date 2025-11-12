#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined plot: Citations vs Cases (Edges vs Nodes) for Australia and the UK on the same axes.
Removes the grey y=x line and the text labels placed next to the lines.
"""

import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
OUTPUTS = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
AUS_XLSX = Path("../data/Completed Australian case set.xlsx")
UK_XLSX  = Path("../data/Aus-time-matched complete UK dataset.xlsx")
PNG_OUT  = OUTPUTS / "AUS vs UK - Citations vs Cases (no-annotations).png"

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
        if s.endswith(".0"): s = s[:-2]
        return s

def parse_ids_from_cell(cell):
    if pd.isna(cell): return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

# ---------- Australia: build summary (Edges vs Nodes over time) ----------
def build_summary_aus(xlsx_path, year_start=1994, year_end=2024):
    df = pd.read_excel(xlsx_path, sheet_name=0, header=0)
    ID_COL, YEAR_COL, CITED_BY = df.columns[0], df.columns[3], df.columns[4]

    df = df.copy()
    df[ID_COL]   = df[ID_COL].apply(clean_id)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[ID_COL].notna()].copy()
    id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

    rows = []
    for year in range(year_start, year_end + 1):
        sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
        valid_ids = set(sub[ID_COL].dropna().tolist())
        n, m = len(valid_ids), 0
        if n >= 1:
            for _, row in sub.iterrows():
                target = row[ID_COL]
                for citer in parse_ids_from_cell(row.get(CITED_BY)):
                    if citer and citer != target and citer in valid_ids:
                        citer_year = id_to_year.get(citer, np.nan)
                        if not pd.isna(citer_year) and citer_year <= year:
                            m += 1
        rows.append({"Year": year, "Nodes": n, "Edges": m})
    return pd.DataFrame(rows)

# ---------- UK: build summary (Edges vs Nodes over time) ----------
def build_summary_uk(xlsx_path, year_start=1995, year_end=2024):
    df = pd.read_excel(xlsx_path, sheet_name=0, header=0)
    year_col = "Year decided" if "Year decided" in df.columns else df.columns[3]
    citation_columns = df.columns[4:].tolist()

    rows = []
    for year in range(year_start, year_end + 1):
        cases_by_year = df[pd.to_numeric(df[year_col], errors="coerce") <= year].copy()
        n, m = len(cases_by_year), 0
        if n >= 1:
            for _, row in cases_by_year.iterrows():
                for col in citation_columns:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip() != "":
                        m += 1
        rows.append({"Year": year, "Nodes": n, "Edges": m})
    return pd.DataFrame(rows)

# ---------- compute ----------
aus_df = build_summary_aus(AUS_XLSX, year_start=1994, year_end=2024)
uk_df  = build_summary_uk(UK_XLSX,  year_start=1995, year_end=2024)

# precompute limits
max_nodes = max(aus_df["Nodes"].max() if not aus_df.empty else 0,
                uk_df["Nodes"].max()  if not uk_df.empty else 0)
max_edges = max(aus_df["Edges"].max() if not aus_df.empty else 0,
                uk_df["Edges"].max()  if not uk_df.empty else 0)

# ---------- combined plot: Edges (y) vs Nodes (x) ----------
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(aus_df["Nodes"], aus_df["Edges"],
        marker='o', linewidth=2.5, markersize=6, alpha=0.85,
        color='tab:purple', label='Australia')

ax.plot(uk_df["Nodes"], uk_df["Edges"],
        marker='s', linewidth=2.5, markersize=6, alpha=0.85,
        color='tab:orange', label='UK')

# Labels & styling
ax.set_xlabel("Number of Cases (Nodes)", fontsize=12)
ax.set_ylabel("Number of Citations (Edges)", fontsize=12)
ax.set_title("Growth of Citations as a Function of Number of Climate Cases: Australia and the UK (1994–2024)",
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

ax.set_xlim(0, max(1, max_nodes) * 1.05)
ax.set_ylim(0, max(1, max_edges) * 1.10)

ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"Saved combined figure -> {PNG_OUT.resolve()}")
