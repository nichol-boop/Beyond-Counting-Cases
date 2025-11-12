#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
DATA_PATH = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS   = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
CSV_OUT   = OUTPUTS / "UK density statistics over time.csv"
PNG_DASH  = OUTPUTS / "UK density statistics over time.png"
PNG_DEN   = OUTPUTS / "UK density statistics over time - density.png"
PNG_NODES = OUTPUTS / "UK density statistics over time - nodes.png"
PNG_EDGES = OUTPUTS / "UK density statistics over time - edges.png"
PNG_SCAT  = OUTPUTS / "UK density statistics over time - density vs nodes.png"

# ---------- load ----------
df = pd.read_excel(DATA_PATH)

# optional: quick sanity prints
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head())

def calculate_network_density_by_year(df):
    """
    Calculate network density for each year from 1995–2024.
    Density = actual edges / possible edges for a directed graph (n*(n-1)).
    NOTE: This follows your original logic (counts any non-empty citation in columns E+).
    """
    yearly_density = {}
    yearly_stats = {}

    for year in range(1995, 2025):
        cases_by_year = df[df['Year decided'] <= year].copy()
        if len(cases_by_year) == 0:
            yearly_density[year] = 0
            yearly_stats[year] = {'nodes': 0, 'edges': 0, 'possible_edges': 0, 'density': 0.0}
            continue

        num_nodes = len(cases_by_year)

        # citation columns: E onward (index 4+)
        citation_columns = df.columns[4:].tolist()

        actual_edges = 0
        for _, row in cases_by_year.iterrows():
            for col in citation_columns:
                if pd.notna(row[col]) and str(row[col]).strip() != '':
                    try:
                        cited_id = str(row[col]).strip()
                        if cited_id and cited_id != 'nan':
                            actual_edges += 1
                    except (ValueError, TypeError):
                        continue

        possible_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 0
        density = (actual_edges / possible_edges) if possible_edges else 0.0

        yearly_density[year] = density
        yearly_stats[year] = {
            'nodes': num_nodes,
            'edges': actual_edges,
            'possible_edges': possible_edges,
            'density': density
        }

    return yearly_density, yearly_stats

# ---------- compute ----------
density_by_year, stats_by_year = calculate_network_density_by_year(df)

summary_data = []
for year in range(1995, 2025):
    s = stats_by_year[year]
    summary_data.append({
        'Year': year,
        'Nodes': s['nodes'],
        'Edges': s['edges'],
        'Possible_Edges': s['possible_edges'],
        'Density': s['density']
    })
summary_df = pd.DataFrame(summary_data)

print("\nNetwork Density by Year (1995–2024):")
print("=" * 60)
print(summary_df.to_string(index=False, float_format='%.6f'))

# ---------- save CSV ----------
summary_df.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"\nSaved CSV -> {CSV_OUT.resolve()}")

# ---------- plots (save all to outputs) ----------
# Individual: Density over time
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(summary_df['Year'], summary_df['Density'], marker='o', linewidth=2, markersize=4)
ax1.set_title('Network Density Over Time'); ax1.set_xlabel('Year'); ax1.set_ylabel('Density'); ax1.grid(alpha=0.3)
fig1.tight_layout(); fig1.savefig(PNG_DEN, dpi=220, bbox_inches='tight'); plt.close(fig1)

# Individual: Nodes over time
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(summary_df['Year'], summary_df['Nodes'], marker='s', color='green', linewidth=2, markersize=4)
ax2.set_title('Number of Cases (Nodes) Over Time'); ax2.set_xlabel('Year'); ax2.set_ylabel('Number of Cases'); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(PNG_NODES, dpi=220, bbox_inches='tight'); plt.close(fig2)

# Individual: Edges over time
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(summary_df['Year'], summary_df['Edges'], marker='^', color='red', linewidth=2, markersize=4)
ax3.set_title('Number of Citations (Edges) Over Time'); ax3.set_xlabel('Year'); ax3.set_ylabel('Number of Citations'); ax3.grid(alpha=0.3)
fig3.tight_layout(); fig3.savefig(PNG_EDGES, dpi=220, bbox_inches='tight'); plt.close(fig3)

# Individual: Density vs Nodes
fig4, ax4 = plt.subplots(figsize=(8,5))
ax4.scatter(summary_df['Nodes'], summary_df['Density'], alpha=0.7, s=30)
ax4.set_title('Density vs Number of Cases'); ax4.set_xlabel('Number of Cases'); ax4.set_ylabel('Density'); ax4.grid(alpha=0.3)
fig4.tight_layout(); fig4.savefig(PNG_SCAT, dpi=220, bbox_inches='tight'); plt.close(fig4)

# Dashboard (2x2 like your notebook)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(summary_df['Year'], summary_df['Density'], marker='o', linewidth=2, markersize=4)
axs[0, 0].set_title('Network Density Over Time'); axs[0, 0].set_xlabel('Year'); axs[0, 0].set_ylabel('Density'); axs[0, 0].grid(alpha=0.3)

axs[0, 1].plot(summary_df['Year'], summary_df['Nodes'], marker='s', color='green', linewidth=2, markersize=4)
axs[0, 1].set_title('Number of Cases (Nodes) Over Time'); axs[0, 1].set_xlabel('Year'); axs[0, 1].set_ylabel('Number of Cases'); axs[0, 1].grid(alpha=0.3)

axs[1, 0].plot(summary_df['Year'], summary_df['Edges'], marker='^', color='red', linewidth=2, markersize=4)
axs[1, 0].set_title('Number of Citations (Edges) Over Time'); axs[1, 0].set_xlabel('Year'); axs[1, 0].set_ylabel('Number of Citations'); axs[1, 0].grid(alpha=0.3)

axs[1, 1].scatter(summary_df['Nodes'], summary_df['Density'], alpha=0.7, s=30)
axs[1, 1].set_title('Density vs Number of Cases'); axs[1, 1].set_xlabel('Number of Cases'); axs[1, 1].set_ylabel('Density'); axs[1, 1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PNG_DASH, dpi=220, bbox_inches='tight'); plt.close(fig)

print("\nSaved figures:")
for p in [PNG_DEN, PNG_NODES, PNG_EDGES, PNG_SCAT, PNG_DASH]:
    print(" -", p.resolve())
