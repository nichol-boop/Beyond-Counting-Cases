#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
OUTPUTS = Path("../outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("../data/Aus-time-matched complete UK dataset.xlsx")
BASE = "Citations vs Cases"
PLOT_PNG = OUTPUTS / f"{BASE}.png"
REPORT_TXT = OUTPUTS / f"{BASE}.txt"

# Prefer precomputed summary if available
SOURCE_CSV = OUTPUTS / "UK density statistics over time.csv"
if SOURCE_CSV.exists():
    summary_df = pd.read_csv(SOURCE_CSV)
else:
    # Fallback: rebuild summary_df from Excel
    df = pd.read_excel(DATA_PATH)
    year_col = "Year decided" if "Year decided" in df.columns else df.columns[3]
    citation_columns = df.columns[4:].tolist()

    rows = []
    for year in range(1995, 2025):
        cases_by_year = df[pd.to_numeric(df[year_col], errors="coerce") <= year].copy()
        n = len(cases_by_year)
        m = 0
        for _, row in cases_by_year.iterrows():
            for col in citation_columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip() != "":
                    m += 1
        possible = n*(n-1) if n > 1 else 0
        dens = (m/possible) if possible else 0.0
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
ax.set_title('Growth of UK Citations as a Function of Number of Climate Cases (1995-2024)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Set limits with some padding
ax.set_xlim(0, summary_df['Nodes'].max() * 1.05)
ax.set_ylim(0, summary_df['Edges'].max() * 1.1)

ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

# Stats/annotations
final_nodes = int(summary_df.loc[summary_df['Year'] == 2024, 'Nodes'].iloc[0])
final_edges = int(summary_df.loc[summary_df['Year'] == 2024, 'Edges'].iloc[0])
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
        year = int(milestone_row['Year'].iloc[0])
        citation_rate = edges / nodes if nodes > 0 else 0
        milestone_data.append((milestone, nodes, edges, year, citation_rate))

lines = []
lines.append("Citations vs Cases Analysis")
lines.append("=" * 50)
lines.append(f"Final counts (2024):")
lines.append(f"  Cases: {final_nodes}")
lines.append(f"  Citations: {final_edges}")
lines.append(f"  Citations per case: {ratio:.2f}" if final_nodes > 0 else "  Citations per case: n/a")
lines.append("")
lines.append("Citation rates at case milestones:")
for milestone, nodes, edges, year, rate in milestone_data:
    lines.append(f"  At ~{milestone} cases ({nodes} cases, year {year}): {edges} citations, rate = {rate:.2f}")
lines.append("")
lines.append(f"First year with cases: {first_year_with_nodes}")
lines.append(f"First year with citations: {first_year_with_edges}")

REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved figure -> {PLOT_PNG.resolve()}")
print(f"Saved report -> {REPORT_TXT.resolve()}")
