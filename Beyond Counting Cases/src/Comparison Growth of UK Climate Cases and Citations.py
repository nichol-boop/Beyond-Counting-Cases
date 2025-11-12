#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
OUTPUTS = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("../data/Aus-time-matched complete UK dataset.xlsx")
BASE = "Comparison Growth of Cases and Citations"
PLOT_PNG = OUTPUTS / f"{BASE}.png"
REPORT_TXT = OUTPUTS / f"{BASE}.txt"

# Prefer precomputed summary if available (from "UK density over time.py")
SOURCE_CSV = OUTPUTS / "UK density statistics over time.csv"

if SOURCE_CSV.exists():
    summary_df = pd.read_csv(SOURCE_CSV)
else:
    # Fallback: rebuild summary_df quickly from Excel with the same logic you used
    df = pd.read_excel(DATA_PATH)
    year_col = "Year decided" if "Year decided" in df.columns else df.columns[3]
    citation_columns = df.columns[4:].tolist()  # E onward
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
        rows.append({"Year": year, "Nodes": n, "Edges": m, "Possible_Edges": possible, "Density": dens})
    summary_df = pd.DataFrame(rows)

# --- Plot: Growth of cases vs citations on one axis ---
fig, ax = plt.subplots(figsize=(12, 8))

(line_nodes,) = ax.plot(summary_df['Year'], summary_df['Nodes'],
                        marker='s', color='tab:blue', linewidth=2.5, markersize=6,
                        label='Cases (Nodes)', alpha=0.8)

(line_edges,) = ax.plot(summary_df['Year'], summary_df['Edges'],
                        marker='^', color='tab:red', linewidth=2.5, markersize=6,
                        label='Citations (Edges)', alpha=0.8)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Growth of Cases and Citations Over Time (1995-2024)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

max_value = max(summary_df['Nodes'].max(), summary_df['Edges'].max())
ax.set_ylim(0, max_value * 1.1 if max_value > 0 else 1.0)

ax.set_xticks(list(range(1995, 2025, 5)))
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

# Stats/annotations
final_nodes = int(summary_df.loc[summary_df['Year'] == 2024, 'Nodes'].iloc[0])
final_edges = int(summary_df.loc[summary_df['Year'] == 2024, 'Edges'].iloc[0])
ratio = (final_edges / final_nodes) if final_nodes > 0 else float('nan')
textstr = f'Citations/Case Ratio: {ratio:.1f}' if final_nodes > 0 else 'Citations/Case Ratio: n/a'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.85, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

fig.tight_layout()
fig.savefig(PLOT_PNG, dpi=220, bbox_inches='tight')
plt.close(fig)

# --- Console + text report with comparison stats ---
first_nodes_row = summary_df[summary_df['Nodes'] > 0].head(1)
first_edges_row = summary_df[summary_df['Edges'] > 0].head(1)

first_year_with_nodes = int(first_nodes_row['Year'].iloc[0]) if not first_nodes_row.empty else None
first_year_with_edges = int(first_edges_row['Year'].iloc[0]) if not first_edges_row.empty else None

first_nodes = int(first_nodes_row['Nodes'].iloc[0]) if not first_nodes_row.empty else 0
first_edges = int(first_edges_row['Edges'].iloc[0]) if not first_edges_row.empty else 0

def cagr(final, start, years):
    if start > 0 and years and years > 0:
        return ((final / start) ** (1/years) - 1) * 100
    return float('nan')

nodes_years_elapsed = 2024 - first_year_with_nodes if first_year_with_nodes is not None else None
edges_years_elapsed = 2024 - first_year_with_edges if first_year_with_edges is not None else None

nodes_growth = cagr(final_nodes, first_nodes, nodes_years_elapsed) if nodes_years_elapsed is not None else float('nan')
edges_growth = cagr(final_edges, first_edges, edges_years_elapsed) if edges_years_elapsed is not None else float('nan')

crossover_data = summary_df[summary_df['Edges'] > summary_df['Nodes']]
crossover_year = int(crossover_data['Year'].iloc[0]) if not crossover_data.empty else None

lines = []
lines.append("Growth Comparison")
lines.append("=" * 40)
lines.append("Final counts (2024):")
lines.append(f"  Cases: {final_nodes}")
lines.append(f"  Citations: {final_edges}")
lines.append(f"  Citations per case: {ratio:.2f}" if final_nodes > 0 else "  Citations per case: n/a")
lines.append("")
lines.append("Compound Annual Growth Rates (CAGR):")
lines.append(f"  Cases: {nodes_growth:.1f}% per year ({first_year_with_nodes}-2024)" if nodes_years_elapsed is not None else "  Cases: n/a")
lines.append(f"  Citations: {edges_growth:.1f}% per year ({first_year_with_edges}-2024)" if edges_years_elapsed is not None else "  Citations: n/a")
lines.append("")
lines.append(f"Citations first exceeded cases in: {crossover_year if crossover_year is not None else 'None'}")

REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved figure -> {PLOT_PNG.resolve()}")
print(f"Saved report -> {REPORT_TXT.resolve()}")
