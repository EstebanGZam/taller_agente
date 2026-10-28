"""Shared offline fixtures for eligibility regression tests."""

import json
import socket

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from secop_agent import eligibility_agent, settings, tender_documents
from secop_agent.batch_eligibility import load_company

KEY = json.loads(settings.ELIGIBILITY_KEY_FILE.read_text())["processes"]
COMPANY = load_company()


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    for name in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "load_environment", lambda: None)

    def block(*args, **kwargs):
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", block)


class FakeModel(BaseChatModel):
    """Local model exercising real LangChain messages, callbacks and graph memory."""

    responses: list
    _position: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self._position]
        self._position += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class FakeStructuredModel(FakeModel):
    def with_structured_output(self, schema, *, method, **kwargs):
        assert method == settings.STRUCTURED_OUTPUT_METHOD
        return self | RunnableLambda(lambda message: schema.model_validate_json(message.content))


def message(text="", tools=None):
    return AIMessage(
        content=text,
        tool_calls=tools or [],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        response_metadata={"model_name": "claude-haiku-5-5"},
    )


def tool_call(name, arguments=None):
    return {"name": name, "args": arguments or {}, "id": "id-" + name, "type": "tool_call"}


def full_requirements(process):
    return {
        **process["expected_requirements"],
        "official_budget": process["official_budget"],
        "reference": process["reference"],
        "entity": process["entity"],
        "modality": process["modality"],
        "purpose": process["reference"],
        "publication_date": "2019-04-29" if process["n"] == 11 else "2024-01-01",
    }


@pytest.fixture
def fake_extraction(monkeypatch):
    def extract(text, model=None, config=None):
        assert config and config["callbacks"]
        return full_requirements(next(process for process in KEY if process["reference"] == text))

    def similar(purpose, experience, model=None, config=None):
        assert config and config["callbacks"]
        return next(
            process["expected_similar_contracts"] for process in KEY if process["reference"] == purpose
        )

    def read(path):
        return next(process["reference"] for process in KEY if process["file"] == path.name)

    monkeypatch.setattr(tender_documents, "extract_requirements", extract)
    monkeypatch.setattr(tender_documents, "select_similar_contracts", similar)
    monkeypatch.setattr(tender_documents, "read_pdf", read)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-offline-key")
    monkeypatch.setattr(
        eligibility_agent,
        "chat_model",
        lambda *args, **kwargs: FakeModel(
            responses=[message("No consulté herramientas."), message("Seguimiento con memoria.")]
        ),
    )
