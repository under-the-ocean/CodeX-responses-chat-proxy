import json

from responses_chat_proxy import launcher


def test_save_and_load_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config = {
        "upstream_base_url": "https://example.com/v1",
        "upstream_api_key": "sk-test",
    }

    launcher.save_config(config_path, config)

    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert launcher.load_config(config_path) == config


def test_load_config_ignores_invalid_json(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{bad json", encoding="utf-8")

    assert launcher.load_config(config_path) == {}


def test_prompt_for_config_uses_default_base_url_and_strips_trailing_slash(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr(launcher.getpass, "getpass", lambda _prompt: " sk-test ")

    config = launcher.prompt_for_config({"upstream_base_url": "https://example.com/v1/"})

    assert config == {
        "upstream_base_url": "https://example.com/v1",
        "upstream_api_key": "sk-test",
    }


def test_apply_runtime_defaults(monkeypatch) -> None:
    for key in ("UPSTREAM_BASE_URL", "UPSTREAM_API_KEY", "PROXY_API_KEY", "HOST", "PORT"):
        monkeypatch.delenv(key, raising=False)

    launcher.apply_runtime_defaults(
        {
            "upstream_base_url": "https://example.com/v1",
            "upstream_api_key": "sk-test",
        }
    )

    assert launcher.os.environ["UPSTREAM_BASE_URL"] == "https://example.com/v1"
    assert launcher.os.environ["UPSTREAM_API_KEY"] == "sk-test"
    assert launcher.os.environ["PROXY_API_KEY"] == ""
    assert launcher.os.environ["HOST"] == "127.0.0.1"
    assert launcher.os.environ["PORT"] == "8000"


def test_should_reconfigure_accepts_enter_to_reuse(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert not launcher.should_reconfigure(
        {
            "upstream_base_url": "https://example.com/v1",
            "upstream_api_key": "sk-test-123456",
        }
    )


def test_mask_api_key() -> None:
    assert launcher.mask_api_key("sk-1234567890") == "sk-1...7890"
    assert launcher.mask_api_key("short") == "*****"
