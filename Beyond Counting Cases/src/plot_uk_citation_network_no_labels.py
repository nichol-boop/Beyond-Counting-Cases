#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def main():
    # ------------------------- locate UK.gexf -------------------------
    candidates = [
        Path("../outputs/UK.gexf"),
        Path("./outputs/UK.gexf"),
        Path("outputs/UK.gexf"),
        Path("/outputs/UK.gexf"),
        Path("UK.gexf"),
    ]
    gexf_path = next((p for p in candidates if p.exists()), None)
    if gexf_path is None:
        raise FileNotFoundError("Could not find UK.gexf in ../outputs, ./outputs, outputs, /outputs or current dir.")
    print(f"Using GEXF: {gexf_path.resolve()}")

    # outputs
    out_dir = Path("../outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "UK Graph No Node Labels.png"

    # ------------------------- load & prep -------------------------
    G = nx.read_gexf(gexf_path)
    in_degree = dict(G.in_degree())

    # years for coloring (expects node attribute 'year')
    years = {}
    for n in G.nodes():
        y = G.nodes[n].get("year")
        years[n] = int(float(y)) if y is not None and str(y).strip() != "" else None
    present = [v for v in years.values() if v is not None]
    fallback_year = int(np.median(present)) if present else 2000
    for n, v in list(years.items()):
        if v is None:
            years[n] = fallback_year

    year_values = [years[n] for n in G.nodes()]
    vmin, vmax = int(min(year_values)), int(max(year_values))
    cmap = plt.cm.viridis

    # ------------------------- layout -------------------------
    print("Computing layout... (this may take ~30–60s on larger graphs)")
    pos = nx.spring_layout(G, k=2.5, iterations=500, seed=42, scale=20)

    # center the largest component
    components = list(nx.weakly_connected_components(G))
    lcc = max(components, key=len)
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    bbox_center = np.array([(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0])
    lcc_centroid = np.array([pos[n] for n in lcc]).mean(axis=0)
    shift = bbox_center - lcc_centroid
    for n in G.nodes():
        pos[n] = (pos[n][0] + shift[0], pos[n][1] + shift[1])

    # push smaller components outward slightly
    for comp in components:
        if comp is lcc:
            continue
        comp_xy = np.array([pos[n] for n in comp])
        comp_centroid = comp_xy.mean(axis=0)
        vec = comp_centroid - bbox_center
        push = 0.15 * vec
        for n in comp:
            pos[n] = (pos[n][0] + push[0], pos[n][1] + push[1])

    # ------------------------- size mapping (emphasize hubs) -------------------------
    nodes_list = list(G.nodes())
    deg_vals = np.array([in_degree[n] for n in nodes_list], dtype=float)
    dmin, dmax = float(deg_vals.min()), float(deg_vals.max())

    min_size, max_size = 120.0, 2400.0  # points^2
    if dmax > dmin:
        norm = (deg_vals - dmin) / (dmax - dmin)
        boosted = norm ** 2.6
        node_sizes = min_size + (max_size - min_size) * boosted
    else:
        node_sizes = np.full_like(deg_vals, min_size)

    # ------------------------- no-overlap relaxation -------------------------
    def sizes_to_radii(sizes, smin=min_size, smax=max_size, rmin=0.26, rmax=1.35):
        sizes = np.asarray(sizes, dtype=float)
        if smax == smin:
            return np.full_like(sizes, (rmin + rmax) / 2.0)
        z = (sizes - smin) / (smax - smin)
        return rmin + (rmax - rmin) * np.clip(z, 0, 1)

    radii = {n: r for n, r in zip(nodes_list, sizes_to_radii(node_sizes))}

    def relax_positions(pos_dict, radii_dict, iterations=220, step=0.7, padding=0.08):
        nodes = list(pos_dict.keys())
        for _ in range(iterations):
            moved = 0
            for i in range(len(nodes)):
                ni = nodes[i]; xi, yi = pos_dict[ni]; ri = radii_dict[ni]
                for j in range(i + 1, len(nodes)):
                    nj = nodes[j]; xj, yj = pos_dict[nj]; rj = radii_dict[nj]
                    dx, dy = xj - xi, yj - yi
                    dist = np.hypot(dx, dy) + 1e-9
                    min_d = ri + rj + padding
                    if dist < min_d:
                        push = (min_d - dist) * 0.5 * step
                        ux, uy = dx / dist, dy / dist
                        xi -= ux * push; yi -= uy * push
                        xj += ux * push; yj += uy * push
                        pos_dict[ni] = (xi, yi); pos_dict[nj] = (xj, yj)
                        moved += 1
            if moved == 0:
                break

    relax_positions(pos, radii, iterations=220, step=0.7, padding=0.08)

    # ------------------------- drawing -------------------------
    fig, ax = plt.subplots(figsize=(20, 16), dpi=140)
    ax.set_facecolor("#ffffff")

    # trim edge lines so they don't run under nodes
    edge_base_color = "#333333"
    edge_base_alpha = 0.40
    lines_x, lines_y = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy) + 1e-9
        ux, uy = dx / L, dy / L
        start_x = x0 + ux * radii[u] * 0.95
        start_y = y0 + uy * radii[u] * 0.95
        end_x   = x1 - ux * radii[v] * 0.95
        end_y   = y1 - uy * radii[v] * 0.95
        lines_x.append([start_x, end_x])
        lines_y.append([start_y, end_y])

    # base edges behind everything
    for (x0, x1), (y0, y1) in zip(lines_x, lines_y):
        ax.plot([x0, x1], [y0, y1], color=edge_base_color, alpha=edge_base_alpha, linewidth=1.15, zorder=1)

    # light open chevrons at midpoints (behind nodes)
    arrow_color = "#bbbbbb"
    arrow_alpha = 0.95
    theta = np.deg2rad(24)

    for (x0, x1), (y0, y1) in zip(lines_x, lines_y):
        seg_dx, seg_dy = x1 - x0, y1 - y0
        seg_L = np.hypot(seg_dx, seg_dy) + 1e-9
        mx = x0 + 0.5 * seg_dx
        my = y0 + 0.5 * seg_dy
        L_arrow = max(0.15, min(0.7, 0.06 * seg_L))
        ux, uy = seg_dx / seg_L, seg_dy / seg_L
        nx_, ny_ = -uy, ux
        w1x = (-ux * np.cos(theta) + nx_ * np.sin(theta))
        w1y = (-uy * np.cos(theta) + ny_ * np.sin(theta))
        w2x = (-ux * np.cos(theta) - nx_ * np.sin(theta))
        w2y = (-uy * np.cos(theta) - ny_ * np.sin(theta))
        xA1, yA1 = mx + L_arrow * w1x, my + L_arrow * w1y
        xA2, yA2 = mx + L_arrow * w2x, my + L_arrow * w2y
        ax.plot([mx, xA1], [my, yA1], color=arrow_color, alpha=arrow_alpha, linewidth=0.9, zorder=2)
        ax.plot([mx, xA2], [my, yA2], color=arrow_color, alpha=arrow_alpha, linewidth=0.9, zorder=2)

    # nodes ON TOP of chevrons (no labels)
    sc = ax.scatter(
        [pos[n][0] for n in nodes_list],
        [pos[n][1] for n in nodes_list],
        c=[years[n] for n in nodes_list],
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=node_sizes, alpha=0.95,
        edgecolors="white", linewidths=2.2, zorder=6
    )

    # title
    ax.set_title("United Kingdom Climate Judgments Citation Network (1995–2024)",
                 fontsize=18, fontweight="bold", pad=14)

    # colorbar with year ticks
    year_range = vmax - vmin
    step = 5 if year_range > 15 else 1
    ticks = list(range(vmin, vmax + 1, step))
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=40)
    cbar.set_label("Decision year", rotation=270, labelpad=25, fontsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

    ax.axis("off")
    plt.tight_layout()

    # -------- save PNG --------
    fig = plt.gcf()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG -> {out_png.resolve()}")

if __name__ == "__main__":
    main()
