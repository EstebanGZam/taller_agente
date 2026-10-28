"""Model construction and environment hygiene shared by every entry point."""

from secop_agent import settings


def test_chat_model_sets_timeout_and_no_sampling_parameters(monkeypatch):
    captured = {}

    def fake_init(model, **kwargs):
        captured.update(model=model, **kwargs)
        return object()

    monkeypatch.setattr(settings, "init_chat_model", fake_init)
    settings.chat_model("anthropic:claude-haiku-5-5", max_tokens=100)
    assert captured["model"] == "anthropic:claude-haiku-5-5"
    assert captured["timeout"] == settings.MODEL_TIMEOUT_S
    assert captured["max_retries"] == settings.MODEL_MAX_RETRIES
    assert not {"temperature", "top_p", "top_k"} & captured.keys()


def test_chat_model_defaults_to_env_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(settings, "init_chat_model", lambda model, **kw: captured.setdefault("model", model))
    monkeypatch.setenv("MODEL", "anthropic:claude-sonnet-5-5")
    settings.chat_model(max_tokens=10)
    assert captured["model"] == "anthropic:claude-sonnet-5-5"


def test_load_environment_unsets_conflicting_langsmith_vars(monkeypatch):
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "x")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "y")
    settings.load_environment()
    import os

    assert "LANGSMITH_AGENT_ID" not in os.environ
    assert "LANGSMITH_AGENT_ENVIRONMENT" not in os.environ
