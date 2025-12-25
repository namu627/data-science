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

# 예시 2: 관심 있는 상위 중심성 노드만 남겨 집중 분석
degree_scores = dict(G_largest.degree())
top_k = 1000  # 유지할 노드 수를 상황에 맞게 조정
top_nodes = sorted(
    degree_scores,
    key=degree_scores.get,
    reverse=True
)[:top_k]
G_focus = G_largest.subgraph(top_nodes).copy()
print_graph_info(G_focus, "Focused Subgraph")
