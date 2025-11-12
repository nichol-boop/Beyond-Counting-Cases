#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ---------- paths ----------
DATA_PATH = Path("../data/Completed Australian case set.xlsx")
OUTPUTS   = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
CSV_OUT   = OUTPUTS / "AUS components over time.csv"
PNG_NUM   = OUTPUTS / "AUS components over time - number.png"
PNG_SHARE = OUTPUTS / "AUS components over time - share largest.png"

# ---------- load ----------
df = pd.read_excel(DATA_PATH)

# Columns: A=0 (ID), D=3 (Year), E=4 (Cited by)
ID_COL = df.columns[0]
YEAR_COL = df.columns[3]
CITED_BY_COL = df.columns[4]

def clean_id(val):
    """Normalise case IDs (12.0 → '12', etc.)"""
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

def parse_ids_from_cell(cell):
    """Extract all integer IDs from the 'Cited by' text field."""
    if pd.isna(cell):
        return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

df = df.copy()
df[ID_COL] = df[ID_COL].apply(clean_id)
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df = df[df[ID_COL].notna()].copy()

id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

def edges_up_to(year: int, allow_same_year=True):
    """Return edges (citer → target) and valid node set for all cases up to a given year."""
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
    valid_ids = set(sub[ID_COL].dropna().tolist())
    E = []
    for _, row in sub.iterrows():
        target = row[ID_COL]
        target_y = row[YEAR_COL]
        if not target:
            continue
        citers = parse_ids_from_cell(row.get(CITED_BY_COL))
        for citer in citers:
            if not citer or citer == target or citer not in valid_ids:
                continue
            citer_y = id_to_year.get(citer, np.nan)
            if pd.isna(citer_y):
                continue
            # forbid future citations
            if allow_same_year:
                if citer_y > year:
                    continue
            else:
                if citer_y >= target_y:
                    continue
            E.append((citer, target))
    return E, valid_ids

# ---------- compute over time ----------
rows = []
for year in range(1994, 2025):
    E, V = edges_up_to(year, allow_same_year=True)
    G = nx.DiGraph(); G.add_nodes_from(V); G.add_edges_from(E)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    weak_all = list(nx.weakly_connected_components(G)) if n else []
    # keep only non-trivial components (size ≥ 2)
    weak_nontriv = [c for c in weak_all if len(c) >= 2]

    num_nontriv = len(weak_nontriv)
    largest_nontriv = max((len(c) for c in weak_nontriv), default=0)
    share_largest_nontriv = (largest_nontriv / n) if (n > 0 and m > 0 and largest_nontriv > 0) else 0.0

    rows.append({
        "Year": year,
        "Nodes": n,
        "Edges": m,
        "WeakComponents_All": len(weak_all),
        "WeakComponents_Nontrivial": num_nontriv,
        "LargestWeakComponent_Nontrivial": largest_nontriv,
        "ShareLargestWeakComponent_Nontrivial": share_largest_nontriv
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"Saved CSV -> {CSV_OUT.resolve()}")

# ---------- plots ----------
# 1) Number of non-trivial components
fig1, ax1 = plt.subplots(figsize=(9, 5.2))
ax1.plot(summary_df["Year"], summary_df["WeakComponents_Nontrivial"],
         marker='o', linewidth=2, markersize=4, color="tab:purple")
ax1.set_title("Number of Weakly Connected Components (size ≥ 2) – Australia")
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of components")
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig(PNG_NUM, dpi=220, bbox_inches='tight')
plt.close(fig1)
print(f"Saved figure -> {PNG_NUM.resolve()}")

# 2) Share of nodes in largest weakly connected component
fig2, ax2 = plt.subplots(figsize=(9, 5.2))
ax2.plot(summary_df["Year"], summary_df["ShareLargestWeakComponent_Nontrivial"],
         marker='s', linewidth=2, markersize=4, color="tab:red")
ax2.set_title("Share of Australian Climate Cases in Largest Weakly Connected Component (1994–2024)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Share of all cases")
ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(PNG_SHARE, dpi=220, bbox_inches='tight')
plt.close(fig2)
print(f"Saved figure -> {PNG_SHARE.resolve()}")
