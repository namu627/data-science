# File: your-workspace/app.py
"""
Dash 애플리케이션의 엔트리 포인트 모듈.

이 모듈은 서버 실행만 담당한다.
실제 비즈니스 로직(데이터 로딩, 레이아웃 구성, 콜백 등록)은 services/ 하위 모듈로 분리되어 있다.
학습 포인트:
- 엔트리 레벨에서는 "조립"만 담당하고, 기능은 모듈로 위임하면 유지보수가 쉬워진다.
"""

from dash import Dash
from config import DATA_PATH, HOST, PORT, DEBUG
from services.data_loader import load_sales_csv
from services.layout import build_layout
from services.callbacks import register_callbacks


def create_app() -> Dash:
    """
    Dash 애플리케이션 인스턴스를 생성하고 구성한다.

    Returns
    -------
    Dash
        레이아웃과 콜백이 모두 세팅된 Dash 앱 인스턴스.
    """
    data_frame = load_sales_csv(DATA_PATH)

    app = Dash(__name__)
    app.title = "Hongik Dashboard"

    # 레이아웃/콜백은 services 모듈로 분리하여 가독성을 높인다.
    app.layout = build_layout(data_frame)
    register_callbacks(app, data_frame)
    return app


if __name__ == "__main__":
    # 개발 모드 실행
    application = create_app()
    application.run(host=HOST, port=PORT, debug=DEBUG)
