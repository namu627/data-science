# 샘플 그래프 생성
# 중심성(PageRank, Betweenness) 계산
# PageRank로 노드 크기, Betweenness로 노드 색상 매핑
# 시각화 및 파일 저장

import matplotlib.pyplot as plt
import networkx as nx

# 1) 샘플 그래프 생성 (무방향)
G = nx.Graph()
edges = [
    (1, 3), (1, 4), (2, 3), (3, 4),   # cluster A
    (4, 5),                            # bridge
    (5, 6), (5, 7), (5, 8),           # cluster B
    (6, 7), (6, 8), (7, 8),
    (7, 9)                             # tail
]
G.add_edges_from(edges)

# 2) 중심성 계산 (필요 지표만 계산)
centralities = {
    "pagerank": nx.pagerank(G, alpha=0.85),
    "betweenness": nx.betweenness_centrality(G, normalized=True),
}

# 3) 레이아웃 및 시각화 매핑
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

# PageRank -> 노드 크기, Betweenness -> 노드 색
sizes = [centralities["pagerank"][node] * 5_000 for node in G.nodes()]
colors = [centralities["betweenness"][node] for node in G.nodes()]

# 4) 그래프 그리기
nx.draw_networkx(
    G,
    pos=pos,
    node_size=sizes,
    node_color=colors,
    cmap="viridis",
    with_labels=True,     # 노드 번호 라벨
    font_size=9,
    font_color="white",   # 어두운 색 노드에서도 보이도록
)

plt.title("Visualization Strategy for Centrality Measures", fontsize=16)
plt.axis("off")
plt.tight_layout()

# 5) 저장 및 표시
plt.savefig("centrality_visualization_strategy.png", dpi=300)
plt.show()
