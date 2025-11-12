#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
OUTPUTS = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("../data/Completed Australian case set.xlsx")  # <-- updated data source
BASE = "AUS Citations vs Cases"
PLOT_PNG = OUTPUTS / f"{BASE}.png"
REPORT_TXT = OUTPUTS / f"{BASE}.txt"

# ---------- helpers ----------
def clean_id(val):
    """Canonical string ID: trims; 12.0 -> '12'; drops empties/NaN."""
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
    """Extract all integer IDs from a free-text cell (e.g., '1, 2; 3\n[4]')."""
    if pd.isna(cell):
        return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]

# ---------- load ----------
df = pd.read_excel(DATA_PATH, sheet_name=0, header=0)

# Column mapping by position (0-based): A=0 (ID), D=3 (Year), E=4 (Cited by)
ID_COL    = df.columns[0]   # Column A
YEAR_COL  = df.columns[3]   # Column D
CITED_BY  = df.columns[4]   # Column E

# Clean / coerce core columns
df = df.copy()
df[ID_COL]   = df[ID_COL].apply(clean_id)
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
# ensure we only keep rows with an ID
df = df[df[ID_COL].notna()].copy()

# ID -> year map for temporal checks
id_to_year = dict(zip(df[ID_COL], df[YEAR_COL]))

# ---------- rebuild summary over time from the Australian Excel ----------
rows = []
for year in range(1995, 2025):
    # valid cases decided up to this cutoff
    sub = df[(df[YEAR_COL].notna()) & (df[YEAR_COL] <= year)].copy()
    valid_ids = set(sub[ID_COL].dropna().tolist())
    n = len(valid_ids)

    if n < 2:
        rows.append({"Year": year, "Nodes": n, "Edges": 0,
                     "Possible_Edges": 0, "Density": 0.0})
        continue

    # count directed edges (citer -> cited/target) where both ends are valid and not future-dated
    m = 0
    for _, row in sub.iterrows():
        target = row[ID_COL]  # the case being cited (this row)
        # parse all citer IDs from Column E
        citers = parse_ids_from_cell(row.get(CITED_BY))
        for citer in citers:
            if citer == target:
                continue
            if citer in valid_ids:
                citer_year = id_to_year.get(citer, np.nan)
                if not pd.isna(citer_year) and citer_year <= year:
                    m += 1

    possible = n * (n - 1)
    dens = (m / possible) if possible else 0.0
    rows.append({"Year": year, "Nodes": n, "Edges": m,
                 "Possible_Edges": possible, "Density": dens})

summary_df = pd.DataFrame(rows)

# --- Plot: Citations vs Cases ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot with cases on x-axis and citations on y-axis
ax.plot(summary_df['Nodes'], summary_df['Edges'],
        marker='o', color='tab:purple', linewidth=2.5,
        markersize=7, alpha=0.7, label='Citations vs Cases')

ax.set_xlabel('Number of Cases', fontsize=12)
ax.set_ylabel('Number of Citations', fontsize=12)
ax.set_title('Growth of Australian Citations as a Function of Number of Climate Cases (1994–2024)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Set limits with some padding
ax.set_xlim(0, max(1, summary_df['Nodes'].max()) * 1.05)
ax.set_ylim(0, max(1, summary_df['Edges'].max()) * 1.1)

ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

# Stats/annotations (final year = 2024)
final_row = summary_df.loc[summary_df['Year'] == 2024]
if not final_row.empty:
    final_nodes = int(final_row['Nodes'].iloc[0])
    final_edges = int(final_row['Edges'].iloc[0])
else:
    final_nodes = int(summary_df['Nodes'].iloc[-1])
    final_edges = int(summary_df['Edges'].iloc[-1])

ratio = (final_edges / final_nodes) if final_nodes > 0 else float('nan')
textstr = f'Final (2024):\nCases: {final_nodes}\nCitations: {final_edges}\nRatio: {ratio:.2f}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.75, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

fig.tight_layout()
fig.savefig(PLOT_PNG, dpi=220, bbox_inches='tight')
plt.close(fig)

# --- Console + text report ---
first_nodes_row = summary_df[summary_df['Nodes'] > 0].head(1)
first_edges_row = summary_df[summary_df['Edges'] > 0].head(1)

first_year_with_nodes = int(first_nodes_row['Year'].iloc[0]) if not first_nodes_row.empty else None
first_year_with_edges = int(first_edges_row['Year'].iloc[0]) if not first_edges_row.empty else None
first_nodes = int(first_nodes_row['Nodes'].iloc[0]) if not first_nodes_row.empty else 0
first_edges = int(first_edges_row['Edges'].iloc[0]) if not first_edges_row.empty else 0

# Calculate citation rate at different case milestones
milestones = [10, 25, 50, 100, 150, 200]
milestone_data = []
for milestone in milestones:
    milestone_row = summary_df[summary_df['Nodes'] >= milestone].head(1)
    if not milestone_row.empty:
        nodes = int(milestone_row['Nodes'].iloc[0])
        edges = int(milestone_row['Edges'].iloc[0])
        year_hit = int(milestone_row['Year'].iloc[0])
        citation_rate = edges / nodes if nodes > 0 else 0
        milestone_data.append((milestone, nodes, edges, year_hit, citation_rate))

lines = []
lines.append("Citations vs Cases Analysis (Australia)")
lines.append("=" * 50)
lines.append("Final counts (2024):")
lines.append(f"  Cases: {final_nodes}")
lines.append(f"  Citations: {final_edges}")
lines.append(f"  Citations per case: {ratio:.2f}" if final_nodes > 0 else "  Citations per case: n/a")
lines.append("")
lines.append("Citation rates at case milestones:")
for milestone, nodes, edges, year_hit, rate in milestone_data:
    lines.append(f"  At ~{milestone} cases ({nodes} cases, year {year_hit}): {edges} citations, rate = {rate:.2f}")
lines.append("")
lines.append(f"First year with cases: {first_year_with_nodes}")
lines.append(f"First year with citations: {first_year_with_edges}")

REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved figure -> {PLOT_PNG.resolve()}")
print(f"Saved report -> {REPORT_TXT.resolve()}")
