from responses_chat_proxy.main import app


def test_app_imports_with_response_return_annotations() -> None:
    routes = {route.path for route in app.routes}

    assert "/v1/responses" in routes
    assert "/v1/chat/completions" in routes
