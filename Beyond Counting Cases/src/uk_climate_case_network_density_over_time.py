#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
DATA_PATH = Path("../data/Aus-time-matched complete UK dataset.xlsx")
OUTPUTS   = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
PNG_DEN   = OUTPUTS / "UK climate case network density over time.png"

# ---------- load ----------
df = pd.read_excel(DATA_PATH)

def calculate_network_density_by_year(df):
    """
    Calculate network density for each year from 1995–2024.
    Density = actual edges / possible edges for a directed graph (n*(n-1)).
    Counts any non-empty citation in columns E onward.
    """
    results = []

    for year in range(1995, 2025):
        cases_by_year = df[df['Year decided'] <= year].copy()
        n = len(cases_by_year)

        if n < 2:
            results.append({"Year": year, "Density": 0.0})
            continue

        citation_columns = df.columns[4:].tolist()
        m = 0  # edge count

        for _, row in cases_by_year.iterrows():
            for col in citation_columns:
                if pd.notna(row[col]) and str(row[col]).strip() not in ("", "nan"):
                    m += 1

        possible_edges = n * (n - 1)
        density = (m / possible_edges) if possible_edges else 0.0
        results.append({"Year": year, "Density": density})

    return pd.DataFrame(results)

# ---------- compute ----------
summary_df = calculate_network_density_by_year(df)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(summary_df["Year"], summary_df["Density"], marker='o', linewidth=2.5, markersize=5, color='tab:orange')
ax.set_title("UK Climate Case Network Density Over Time", fontsize=14, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Network Density", fontsize=12)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PNG_DEN, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f"Saved figure -> {PNG_DEN.resolve()}")
