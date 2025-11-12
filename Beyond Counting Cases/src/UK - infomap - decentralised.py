
import math, networkx as nx

def community_layout(G, community_attr="module", seed=42,
                     k_comm=5.0, k_intra=0.3, scale_comm=5.0, intra_scale=0.6):
    mods = nx.get_node_attributes(G, community_attr)
    comms = {}
    for n, m in mods.items():
        comms.setdefault(m, []).append(n)

    H = nx.Graph()
    for m, nodes in comms.items():
        H.add_node(m, size=len(nodes))
    for u, v in G.edges():
        mu, mv = mods.get(u), mods.get(v)
        if mu is not None and mv is not None and mu != mv:
            H.add_edge(mu, mv, weight=H.get_edge_data(mu, mv, {}).get("weight", 0) + 1)

    if H.number_of_nodes() == 0:
        pos_comm = {list(comms.keys())[0]: (0.0, 0.0)}
    else:
        pos_comm = nx.spring_layout(H, k=k_comm, weight="weight", seed=seed, scale=scale_comm)

    pos = {}
    for m, nodes in comms.items():
        S = G.subgraph(nodes)
        sub = {nodes[0]: (0.0, 0.0)} if S.number_of_nodes()==1 else nx.spring_layout(S, k=k_intra, seed=seed, scale=1.0)
        cx, cy = pos_comm.get(m, (0.0, 0.0))
        r = intra_scale * math.sqrt(len(nodes))
        for n, (x, y) in sub.items():
            pos[n] = (cx + r*x, cy + r*y)
    return pos, comms
