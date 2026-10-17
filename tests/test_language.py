"""Multilingual and code-mixed language detection tests."""

from app.language import LanguageDecision, detect_language_async


async def test_code_mixed_message_uses_language_model_adapter(monkeypatch):
    """Romanized code-mixed input is resolved through the shared LLM adapter."""
    calls = []

    async def detect(task, payload, response_model, **kwargs):
        calls.append((task, payload))
        return LanguageDecision(language="hi")

    monkeypatch.setattr("app.language.complete_json", detect)
    result = await detect_language_async("Main shower le sakta hoon?")
    assert result == "hi"
    assert calls and calls[0][1]["message"] == "Main shower le sakta hoon?"


async def test_native_scripts_are_detected_without_llm(monkeypatch):
    """Native Hindi and Telugu scripts use deterministic script detection."""

    async def should_not_call(*args, **kwargs):
        raise AssertionError("Native script detection should not call the LLM")

    monkeypatch.setattr("app.language.complete_json", should_not_call)
    assert await detect_language_async("నాకు నీళ్లు కావాలి") == "te"
    assert await detect_language_async("मुझे पानी चाहिए") == "hi"
