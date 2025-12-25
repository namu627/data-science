import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# 1) Base graph
G = nx.Graph()
G.add_edges_from([
    (1, 3), (1, 4), (2, 3), (3, 4),
    (4, 5),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8), (7, 8),
    (7, 9)
])

# 2) Augment graph: duplicate with +10 offset and add cross-edges
H = nx.Graph()
H.add_edges_from(G.edges())
H.add_edges_from([(u + 10, v + 10) for u, v in G.edges()])
G_aug = nx.compose(G, H)
G_aug.add_edges_from([(4, 14), (8, 18), (6, 17)])

# 3) Centralities
deg   = nx.degree_centrality(G_aug)
bet   = nx.betweenness_centrality(G_aug)
close = nx.closeness_centrality(G_aug)
eig   = nx.eigenvector_centrality(G_aug, max_iter=500)
pr    = nx.pagerank(G_aug, alpha=0.85)

# 4) Nodes that achieve the maximum value per centrality (for red highlight)
max_deg_nodes   = [n for n, v in deg.items()   if v == max(deg.values())]
max_bet_nodes   = [n for n, v in bet.items()   if v == max(bet.values())]
max_close_nodes = [n for n, v in close.items() if v == max(close.values())]
max_eig_nodes   = [n for n, v in eig.items()   if v == max(eig.values())]
max_pr_nodes    = [n for n, v in pr.items()    if v == max(pr.values())]

# 5) Layout
pos = nx.spring_layout(G_aug, seed=42)

# 6) Figure (no global title)
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.0])
ax_graph = fig.add_subplot(gs[0, 0])
ax_info  = fig.add_subplot(gs[0, 1])

# 7) Draw graph
node_sizes  = [pr[n] * 10000 for n in G_aug.nodes()]   # larger nodes
node_colors = [bet[n] for n in G_aug.nodes()]

nx.draw_networkx_edges(G_aug, pos, ax=ax_graph, alpha=0.75, width=2.0)
nx.draw_networkx_nodes(
    G_aug, pos, ax=ax_graph,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="viridis",
    edgecolors="black",
    linewidths=1.5
)

# 8) Node ID labels: white text with thin black outline
for n, (x, y) in pos.items():
    txt = ax_graph.text(
        x, y, str(n),
        fontsize=12, weight="bold",
        ha="center", va="center",
        color="white"
    )
    txt.set_path_effects([
        path_effects.Stroke(linewidth=1.0, foreground="black"),
        path_effects.Normal()
    ])

# 9) Graph title (left axis), no global figure title
ax_graph.set_title("Network Graph (Randomly Generated)", fontsize=14)
ax_graph.axis("off")

# 10) Side panel (right axis)
ax_info.set_title("Centrality Values", fontsize=14)
ax_info.axis("off")

headers = ["Node", "Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"]
x_positions = [0.02, 0.23, 0.45, 0.67, 0.89, 1.11]

# Header row
for x, h in zip(x_positions, headers):
    ax_info.text(x, 0.98, h, fontsize=11, fontweight="bold", family="monospace")

# Rows: nodes ascending
sorted_nodes = sorted(G_aug.nodes())
y_start, dy = 0.94, 0.045

for i, n in enumerate(sorted_nodes):
    y = y_start - i * dy
    if y < 0.02:
        break

    # Determine highlight color for each centrality column
    colors = ["black"] * 5
    if n in max_deg_nodes:   colors[0] = "red"
    if n in max_bet_nodes:   colors[1] = "red"
    if n in max_close_nodes: colors[2] = "red"
    if n in max_eig_nodes:   colors[3] = "red"
    if n in max_pr_nodes:    colors[4] = "red"

    vals = [
        f"{n:>2}",
        f"{deg[n]:.2f}",
        f"{bet[n]:.2f}",
        f"{close[n]:.2f}",
        f"{eig[n]:.2f}",
        f"{pr[n]:.2f}"
    ]

    ax_info.text(x_positions[0], y, vals[0], fontsize=10, family="monospace")
    for idx, v in enumerate(vals[1:], start=1):
        ax_info.text(x_positions[idx], y, v, fontsize=10, family="monospace", color=colors[idx-1])

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 11) Save
plt.savefig("centrality_comparison.png", dpi=260)
plt.show()
