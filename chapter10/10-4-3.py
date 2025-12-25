# interactive_network_3x_exact_fixed.py
# - 단일 그래프에서 노드/엣지 수를 정확히 3배(노드=27, 엣지=36)
# - 중심성 계산 후 pyvis로 인터랙티브 HTML 생성
# - cdn_resources='in_line' 으로 오프라인/방화벽 환경에서도 표시 보장
"""
실습 흐름 가이드
1. PyVis/NetworkX를 불러온 뒤 기본 그래프를 정의한다.
2. 목표 노드·엣지 수에 맞춰 허브 노드를 중심으로 네트워크를 확장한다.
3. 핵심 중심성 지표(PageRank, Betweenness 등)를 계산하고 시각화 매핑용 값을 만든다.
4. PyVis `Network` 객체를 생성해 물리 엔진 옵션, 노드/엣지 스타일을 설정한다.
5. 노드/엣지를 PyVis에 주입하면서 Tooltip, 크기, 색상을 지표와 연결한다.
6. 최종 HTML을 생성·저장하여 브라우저나 리포트에 임베딩한다.
"""

from pyvis.network import Network   # vis.js 기반 인터랙티브 그래프 생성기
import networkx as nx               # 그래프 생성·분석 도구
import math                         # 노드 크기 스케일링에 사용

# 0) Base graph (nodes=9, edges=12) -------------------------------------------
base_edges = [
    (1, 3), (1, 4), (2, 3), (3, 4),
    (4, 5),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (7, 8),
    (7, 9)
]
G = nx.Graph()                      # 무방향 그래프 객체 생성
G.add_edges_from(base_edges)        # 엣지 리스트를 통해 기본 구조 입력

TARGET_NODES = 27                   # 최종 목표 노드 수
TARGET_EDGES = 36                   # 최종 목표 엣지 수

# 1) 노드 개수 채우기 (10..27 추가) + 허브(5)와 연결 -------------------------
current_max = max(G.nodes)          # 현재 그래프의 최대 노드 번호
for new_n in range(current_max + 1, TARGET_NODES + 1):
    G.add_node(new_n)               # 새 노드 추가
    G.add_edge(5, new_n)            # 허브(5)에 연결해 단일 컴포넌트 유지

assert len(G.nodes) == 27, f"nodes={len(G.nodes)} (expect 27)"

# 2) 엣지 수 맞추기 -----------------------------------------------------------
#    - 실습 목표(노드=27, 엣지=36)에 맞춰 정해진 규모를 확보해야
#      이후 중심성 비교나 시각화가 동일 조건에서 재현 가능하다.
#    - 고르게 분포된 추가 엣지를 넣어 그래프가 너무 희박해지지 않도록 한다.
needed = TARGET_EDGES - G.number_of_edges()  # 추가로 필요한 엣지 개수
extra_pairs = [
    (6, 10), (7, 11), (8, 12),
    (15, 16), (20, 21), (22, 23),
    (24, 25), (26, 27)  # 여유 후보
]
added = 0                                # 실제로 추가된 엣지 카운터
for u, v in extra_pairs:
    if added >= needed:
        break                            # 목표 수를 채우면 조기 종료
    if u in G and v in G and not G.has_edge(u, v):
        G.add_edge(u, v)                 # 존재하는 노드 쌍만 연결
        added += 1                       # 새 엣지를 추가했음을 기록

assert G.number_of_edges() == 36, f"edges={G.number_of_edges()} (expect 36)"

# 3) 중심성 계산 --------------------------------------------------------------
deg   = nx.degree_centrality(G)                    # 연결 수 기반 중요도
bet   = nx.betweenness_centrality(G, normalized=True)  # 경로 중개 빈도
close = nx.closeness_centrality(G)                 # 평균 거리 역수
eig   = nx.eigenvector_centrality(G, max_iter=500) # 영향력 높은 이웃과의 연결
pr    = nx.pagerank(G, alpha=0.85)                 # PageRank 확률 분포

"""
중심성 값 활용 팁
- Degree: 연결 허브를 찾는 지표. 보고서에서는 상위 노드를 표로 정리하거나 굵은 테두리로 강조하기 좋다.
- Betweenness: 여러 노드가 거쳐 가는 중개자를 알려주므로 색상(아래 bet_to_hex)으로 매핑해 구조적 브리지를 시각적으로 드러낸다.
- Closeness: 전체를 빠르게 탐색할 수 있는 노드를 보여주니 Tooltip에 포함해 “접근성이 높은 노드”를 설명한다.
- Eigenvector: 영향력 있는 이웃과 연결 여부를 나타내므로 상위 노드를 필터링하거나 추천 후보를 고를 때 활용한다.
- PageRank: 이 스크립트에서는 `scale_pr` 함수에 들어가 PyVis 노드 크기를 결정한다. 확률이 높은 노드일수록 화면에서 크게 보이도록 설정.
"""

# 4) 색상 매핑 (Betweenness → 색) --------------------------------------------
# - betweenness 값이 높을수록 밝은/따뜻한 색으로 보여 중개자가 눈에 띄도록 한다.
# - matplotlib이 설치돼 있으면 Viridis 컬러맵을 사용하고, 설치돼 있지 않은 환경을 대비해
#   try/except로 수동 색상 팔레트(보라→노랑)를 준비해 둔다.
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    def bet_to_hex(betweenness_value: float) -> str:
        norm = mcolors.Normalize(vmin=0, vmax=1)    # 0~1 구간으로 정규화
        return mcolors.to_hex(cm.viridis(norm(betweenness_value)))  # viridis 색상표 사용
except Exception:
    # (Fallback) matplotlib 미설치 또는 노트북 환경 문제 시에도 실행이 끊기지 않도록
    #   미리 정의한 다섯 단계 색상을 사용한다.
    def bet_to_hex(betweenness_value: float) -> str:
        if betweenness_value < 0.2:  return "#440154"
        if betweenness_value < 0.4:  return "#31688e"
        if betweenness_value < 0.6:  return "#35b779"
        if betweenness_value < 0.8:  return "#fde725"
        return "#ffcc00"

# 5) pyvis 네트워크 구성 (중요: cdn_resources='in_line') ----------------------
net = Network(
    height="650px",
    width="100%",
    notebook=False,
    directed=False,
    bgcolor="#ffffff",
    font_color="#222222",
    cdn_resources="in_line"  # 외부 CDN 없이 HTML 하나로 동작하게 설정
)

# vis.js 옵션(JSON)으로 물리 엔진·인터랙션 세부 설정
#   - interaction.hover: 노드 위로 마우스를 가져가면 Tooltip 표시
#   - interaction.navigationButtons: 확대/축소, 중앙 정렬 버튼 UI 제공
#   - interaction.multiselect: 드래그로 여러 노드를 동시에 선택 가능
#   - physics.enabled: 물리 레이아웃 활성화 여부
#   - physics.stabilization: 레이아웃 안정화 반복 횟수 (흔들림을 줄이기 위함)
#   - physics.barnesHut: force-directed 파라미터(중력, 스프링 길이·강도, 감쇠 비율)
#   - nodes.borderWidth / shadow: 노드 테두리 두께와 그림자 여부
#   - edges.smooth: 엣지를 곡선으로 표현해 겹침을 줄임
#   - edges.color.opacity: 엣지 투명도 조절로 배경 대비 확보
net.set_options("""
{
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "multiselect": true
  },
  "physics": {
    "enabled": true,
    "stabilization": {
      "enabled": true,
      "iterations": 400
    },
    "barnesHut": {
      "gravitationalConstant": -25000,
      "centralGravity": 0.3,
      "springLength": 120,
      "springConstant": 0.02,
      "damping": 0.09
    }
  },
  "nodes": {
    "borderWidth": 1,
    "shadow": true
  },
  "edges": {
    "smooth": {
      "type": "dynamic",
      "roundness": 0.4
    },
    "color": {
      "opacity": 0.7
    }
  }
}
""")

# 6) (핵심) 그래프를 pyvis에 실제로 추가 --------------------------------------
#    1) PageRank 값을 노드 크기로 변환할 스케일 함수 정의
#    2) 각 노드에 Tooltip/크기/색상/레이블을 부여하며 PyVis에 add_node
#    3) 모든 엣지를 add_edge로 연결해 PyVis 그래프를 완성

pagerank_values = list(pr.values())                # 노드 크기 스케일링을 위한 값

pagerank_min, pagerank_max = min(pagerank_values), max(pagerank_values)

def scale_pagerank_to_size(
    pagerank_value: float,
    min_size_px: int = 12,
    max_size_px: int = 42
) -> float:
    """
    PageRank 확률(0~1)을 시각적으로 구분 가능한 픽셀 크기로 변환한다.
    - 중심성이 모두 같으면 중간 크기를 고정으로 반환해 레이아웃 붕괴를 방지한다.
    - 그렇지 않으면 선형 보간법으로 최소/최대 크기 사이 값을 돌려준다.
    """
    if math.isclose(pagerank_max, pagerank_min):
        return (min_size_px + max_size_px) / 2   # 모든 값이 같으면 평균 크기
    normalized = (pagerank_value - pagerank_min) / (pagerank_max - pagerank_min)
    return min_size_px + normalized * (max_size_px - min_size_px)  # 원하는 픽셀 범위에 매핑

# 각 노드를 순회하며 PyVis 그래프에 속성 정보를 포함해 추가
#   PyVis에서는 node `title` 속성이 HTML Tooltip으로 노출되므로, 
#   브라우저에서 결과 HTML을 열고
#   노드 위에 마우스를 올리면 중심성 지표가 바로 확인된다. 
#   size/color 속성은 시각적 강조에 즉시 반영된다.
for node_id in G.nodes():
    title = (
        f"<b>Node {node_id}</b><br>"
        f"Degree: {deg[node_id]:.3f}<br>"
        f"Betweenness: {bet[node_id]:.3f}<br>"
        f"Closeness: {close[node_id]:.3f}<br>"
        f"Eigenvector: {eig[node_id]:.3f}<br>"
        f"PageRank: {pr[node_id]:.5f}"
    )
    # 전략: 
    #   label은 읽기 쉬운 문자열, 
    #   title에는 상세 지표를 HTML로, 
    #   size/color는 중심성에 따라 동적으로 매핑
    net.add_node(
        node_id,
        label=str(node_id),
        title=title,                                # 마우스오버 시 표시할 중심성 정보
        size=scale_pagerank_to_size(pr[node_id]),   # PageRank 기반 노드 크기
        color=bet_to_hex(bet[node_id]),             # Betweenness 기반 색상
        borderWidth=1
    )

for source_node, target_node in G.edges():
    net.add_edge(source_node, target_node)  # 노드 간 연결선을 pyvis에 주입

# 7) HTML 생성 (브라우저 자동 열기 X) -----------------------------------------
out_path = "interactive_network.html"   # 결과 파일 경로

# 7-1) pyvis가 만든 HTML 문자열 생성
html = net.generate_html(notebook=False)  # cdn_resources='in_line' 설정 유지한 상태여야 함

# 7-2) UTF-8로 직접 저장
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved: {out_path}")
