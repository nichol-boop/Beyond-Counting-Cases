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
AUS_PATH = Path("../data/Completed Australian case set.xlsx")
UK_PATH  = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS  = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)

PNG_NUM   = OUTPUTS / "AUS vs UK - components number.png"
PNG_SHARE = OUTPUTS / "AUS vs UK - components share.png"

# ---------- helper functions ----------
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
        return s.replace("\u200b", "").strip()

def parse_ids_from_cell(cell):
    if pd.isna(cell): return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

def build_summary_aus(path):
    df = pd.read_excel(path)
    ID_COL, YEAR_COL, CITED_BY_COL = df.columns[0], df.columns[3], df.columns[4]
    df[ID_COL] = df[ID_COL].apply(clean_id)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df[df[ID_COL].notna()].copy()
    id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

    def edges_up_to(year, allow_same_year=True):
        sub = df[df[YEAR_COL] <= year].copy()
        valid_ids = set(sub[ID_COL].dropna().tolist())
        E = []
        for _, row in sub.iterrows():
            target = row[ID_COL]
            for citer in parse_ids_from_cell(row.get(CITED_BY_COL)):
                if not citer or citer == target or citer not in valid_ids: continue
                citer_year = id_to_year.get(citer, np.nan)
                if pd.isna(citer_year): continue
                if allow_same_year:
                    if citer_year > year: continue
                else:
                    if citer_year >= row[YEAR_COL]: continue
                E.append((citer, target))
        return E, valid_ids

    rows = []
    for year in range(1994, 2025):
        E, V = edges_up_to(year)
        G = nx.DiGraph(); G.add_nodes_from(V); G.add_edges_from(E)
        n, m = G.number_of_nodes(), G.number_of_edges()
        weak_all = list(nx.weakly_connected_components(G)) if n else []
        weak_nontriv = [c for c in weak_all if len(c) >= 2]
        num_nontriv = len(weak_nontriv)
        largest_nontriv = max((len(c) for c in weak_nontriv), default=0)
        share_largest = (largest_nontriv / n) if (n > 0 and m > 0 and largest_nontriv > 0) else 0
        rows.append({
            "Year": year,
            "WeakComponents_Nontrivial": num_nontriv,
            "ShareLargestWeakComponent_Nontrivial": share_largest
        })
    return pd.DataFrame(rows)

def build_summary_uk(path):
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

    def edges_up_to(year, allow_same_year=True):
        sub = df[df[YEAR_COL] <= year].copy()
        valid_ids = set(sub[ID_COL].dropna().tolist())
        E = []
        for _, row in sub.iterrows():
            src = row[ID_COL]
            for col in CITATION_COLS:
                tgt = row[col]
                if not tgt or tgt not in valid_ids or tgt == src: continue
                tgt_y = id_to_year.get(tgt, np.nan)
                if pd.isna(tgt_y): continue
                if allow_same_year:
                    if tgt_y > year: continue
                else:
                    if tgt_y >= row[YEAR_COL]: continue
                E.append((src, tgt))
        return E, valid_ids

    rows = []
    for year in range(1995, 2025):
        E, V = edges_up_to(year)
        G = nx.DiGraph(); G.add_nodes_from(V); G.add_edges_from(E)
        n, m = G.number_of_nodes(), G.number_of_edges()
        weak_all = list(nx.weakly_connected_components(G)) if n else []
        weak_nontriv = [c for c in weak_all if len(c) >= 2]
        num_nontriv = len(weak_nontriv)
        largest_nontriv = max((len(c) for c in weak_nontriv), default=0)
        share_largest = (largest_nontriv / n) if (n > 0 and m > 0 and largest_nontriv > 0) else 0
        rows.append({
            "Year": year,
            "WeakComponents_Nontrivial": num_nontriv,
            "ShareLargestWeakComponent_Nontrivial": share_largest
        })
    return pd.DataFrame(rows)

# ---------- compute ----------
aus_df = build_summary_aus(AUS_PATH)
uk_df  = build_summary_uk(UK_PATH)

# ---------- plot 1: number of non-trivial components ----------
fig1, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.plot(aus_df["Year"], aus_df["WeakComponents_Nontrivial"],
         marker='o', color='tab:purple', linewidth=2.5, markersize=6, label='Australia')
ax1.plot(uk_df["Year"], uk_df["WeakComponents_Nontrivial"],
         marker='s', color='tab:orange', linewidth=2.5, markersize=6, label='UK')
ax1.set_title("Number of Weakly Connected Components (size ≥ 2): Australia vs UK (1994–2024)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Year"); ax1.set_ylabel("Number of components")
ax1.grid(alpha=0.3); ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)
fig1.tight_layout(); fig1.savefig(PNG_NUM, dpi=220, bbox_inches='tight'); plt.close(fig1)
print(f"Saved combined figure -> {PNG_NUM.resolve()}")

# ---------- plot 2: share of largest component ----------
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
ax2.plot(aus_df["Year"], aus_df["ShareLargestWeakComponent_Nontrivial"],
         marker='o', color='tab:purple', linewidth=2.5, markersize=6, label='Australia')
ax2.plot(uk_df["Year"], uk_df["ShareLargestWeakComponent_Nontrivial"],
         marker='s', color='tab:orange', linewidth=2.5, markersize=6, label='UK')
ax2.set_title("Share of Climate Cases in Largest Weakly Connected Component: Australia and the UK (1994–2024)", fontsize=13, fontweight='bold')
ax2.set_xlabel("Year"); ax2.set_ylabel("Share of all cases")
ax2.set_ylim(0, 1); ax2.grid(alpha=0.3); ax2.legend(loc="upper left", framealpha=0.9, fontsize=10)
fig2.tight_layout(); fig2.savefig(PNG_SHARE, dpi=220, bbox_inches='tight'); plt.close(fig2)
print(f"Saved combined figure -> {PNG_SHARE.resolve()}")
