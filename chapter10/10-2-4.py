import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import mmread

def print_graph_info(graph, info: str = "Graph"):
    """그래프의 기본 정보를 출력하는 유틸리티 함수"""
    print(f"\n---- {info} ----")
    print(f"Graph Type: {graph.__class__.__name__}")
    print(f"Number of nodes: {graph.number_of_nodes():,}")
    print(f"Number of edges: {graph.number_of_edges():,}")

# Matrix Market 형식(.mtx) 파일을 희소 행렬로 읽어온다.
#  .tocsr: 불러온 그래프를 압축된 희소 행렬(Compressed Sparse Row format)로 변환하는 메서드
matrix = mmread("email-Enron.mtx").tocsr()

# 방향 그래프로 변환한다. (무방향 그래프가 필요하면 nx.Graph() 사용)
G: nx.DiGraph = nx.from_scipy_sparse_array(
    matrix,
    create_using=nx.DiGraph()
)

print_graph_info(G, "Original Graph")

# 네트워크가 너무 크면, 우선 분석하기 좋은 크기의 부분 그래프를 만든다.
# 예시 1: 최대 연결 성분만 추출 (비연결 그래프라면 가장 큰 연결 덩어리만 사용)
largest_component_nodes = max(nx.weakly_connected_components(G), key=len)
G_largest = G.subgraph(largest_component_nodes).copy()
print_graph_info(G_largest, "Largest Connected Component")

# 상위 중심성 노드를 여러 크기로 추출해 비교
degree_scores = dict(G_largest.degree())
sorted_nodes = sorted(
    degree_scores,
    key=degree_scores.get,
    reverse=True
)

subgraph_specs = [
    ("Top-50 Degree Nodes", 50),
    ("Top-10 Degree Nodes", 10),
]

subgraphs = []
for label, k in subgraph_specs:
    target_nodes = sorted_nodes[:min(k, len(sorted_nodes))]
    subgraph = G_largest.subgraph(target_nodes).copy()
    print_graph_info(subgraph, label)
    subgraphs.append((label, subgraph))

fig, axes = plt.subplots(1, len(subgraphs), figsize=(12, 6))

for ax, (label, subgraph) in zip(axes, subgraphs):
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=20, node_color="#1f78b4", alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, width=0.2, alpha=0.4)
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()
# plt.savefig("./subgraph_comparison.png", dpi=300)
# plt.close(fig)
plt.show()
