"""NIM integration checks with simulated streams; no NVIDIA requests."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types import CompletionUsage

import run_nim_evidence as nim
from evidence_preparation import prepare_evidence
from nim_inference import stream_nim_completion
from test_evidence_followup import saved_run


def chunk(content=None, reasoning=None, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, reasoning_content=reasoning),
        finish_reason=finish,
    )])


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.close = Mock()

    def __iter__(self):
        for item in self.chunks:
            if isinstance(item, Exception):
                raise item
            yield item


def fixture_response():
    prepared = prepare_evidence(saved_run())
    group = prepared.groups[0]
    catalog = json.loads(prepared.model_inputs[0].messages[1]["content"])["catalog"]
    reason = next(key for key, unit in catalog.items() if unit["json_pointer"] == "/trialWhyStopped")
    response = {
        "claims": [{"question_id": "Q1", "evidence_ids": [reason]}],
        "review_flags": [{"question_id": "Q1", "evidence_ids": [], "missing_material": "具體停試公告原文"}],
    }
    return prepared, response


@pytest.mark.parametrize("placement", ["none", "with_choice", "usage_only"])
def test_stream_retains_reported_usage_without_requesting_new_parameters(placement):
    expected = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    usage = CompletionUsage(**expected)
    response = chunk("{}", finish="stop")
    chunks = [response]
    if placement == "with_choice":
        response.usage = usage
    elif placement == "usage_only":
        chunks.append(SimpleNamespace(choices=[], usage=usage))
    client = Mock()
    client.chat.completions.create.return_value = FakeStream(chunks)
    messages = [{"role": "user", "content": "offline test"}]
    result = stream_nim_completion(client, messages)
    assert result["status"] == "response_received"
    assert result["response_text"] == "{}"
    assert result["usage"] == (None if placement == "none" else usage.model_dump(mode="json"))
    client.chat.completions.create.assert_called_once_with(
        model=nim.MODEL, messages=messages, **nim.GENERATION_PARAMETERS
    )


def test_one_request_uses_official_parameters_and_separates_reasoning_from_json():
    prepared, response = fixture_response()
    text = json.dumps(response, ensure_ascii=False)
    stream = FakeStream([SimpleNamespace(choices=[]), chunk(reasoning="推理內容"),
                         chunk(text[:30]), chunk(text[30:]), chunk(finish="stop")])
    client = Mock()
    client.chat.completions.create.return_value = stream
    result = nim.run_prepared_group(prepared, prepared.groups[0].group_id, client)
    client.chat.completions.create.assert_called_once_with(
        model=nim.MODEL, messages=prepared.model_inputs[0].messages,
        **nim.GENERATION_PARAMETERS,
    )
    assert result["reasoning_text"] == "推理內容"
    assert result["response_text"] == text
    fact = next(iter(result["structured_output"]["source_facts"].values()))
    assert fact["value"] == "Sponsor's Decision"
    assert result["structured_output"]["claims"] == [{**response["claims"][0], "source_form": "source_field"}]
    gap = result["structured_output"]["review_flags"][0]
    assert gap["missing_material"] == response["review_flags"][0]["missing_material"]
    assert gap["subject_ref"] == "NCT04241068"
    assert result["structured_output"]["source_facts"][gap["evidence_ids"][0]]["value"] == "nct04241068"
    assert response["review_flags"][0]["evidence_ids"] == []
    assert result["structured_output"]["group_id"] == prepared.groups[0].group_id
    assert result["structured_output"]["questions"] == {"Q1": prepared.research_question}
    assert result["status"] == "references_validated"
    assert result["validation_scope"] == "source_assembly_and_references_only"
    stream.close.assert_called_once()


@pytest.mark.parametrize("finish", ["length", None])
def test_truncated_or_interrupted_stream_cannot_be_marked_valid(finish):
    prepared, response = fixture_response()
    client = Mock()
    client.chat.completions.create.return_value = FakeStream([
        chunk(json.dumps(response)), chunk(finish=finish),
    ])
    result = nim.run_prepared_group(prepared, prepared.groups[0].group_id, client)
    assert result["status"] == "incomplete_response"
    assert result["structured_output"] is None
    assert result["response_text"]


@pytest.mark.parametrize("kind", ["invalid_json", "wrong_reference"])
def test_invalid_answer_is_saved_for_diagnosis_without_repair_requests(kind):
    prepared, response = fixture_response()
    if kind == "wrong_reference":
        response["claims"][0]["evidence_ids"] = ["not-supplied"]
    text = "not JSON" if kind == "invalid_json" else json.dumps(response)
    client = Mock()
    client.chat.completions.create.return_value = FakeStream([chunk(text), chunk(finish="stop")])
    result = nim.run_prepared_group(prepared, prepared.groups[0].group_id, client)
    assert result["status"] == "invalid_response"
    assert result["response_text"] == text
    assert result["structured_output"] is None
    assert client.chat.completions.create.call_count == 1


def test_stream_error_preserves_partial_response_and_does_not_log_exception_secrets():
    prepared, _ = fixture_response()
    error = RuntimeError("fake-secret-key-in-network-exception")
    error.status_code = 403
    client = Mock()
    stream = FakeStream([chunk("partial"), error])
    client.chat.completions.create.return_value = stream
    result = nim.run_prepared_group(prepared, prepared.groups[0].group_id, client)
    assert result["status"] == "api_error"
    assert result["error"] == {"type": "RuntimeError", "http_status": 403}
    assert result["response_text"] == "partial"
    assert "fake-secret-key" not in json.dumps(result)
    stream.close.assert_called_once()


def test_markdown_json_fence_is_accepted_but_unknown_group_never_calls_model():
    prepared, response = fixture_response()
    client = Mock()
    client.chat.completions.create.return_value = FakeStream([
        chunk("```json\n" + json.dumps(response) + "\n```"), chunk(finish="stop"),
    ])
    assert nim.run_prepared_group(prepared, prepared.groups[0].group_id, client)["status"] == "references_validated"
    client.reset_mock()
    with pytest.raises(ValueError, match="exactly one"):
        nim.run_prepared_group(prepared, "unknown", client)
    client.chat.completions.create.assert_not_called()


def test_empty_api_key_stops_before_creating_client(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setattr(nim, "load_dotenv", Mock())
    with pytest.raises(SystemExit, match="Fill NVIDIA_API_KEY"):
        nim.main()


def test_main_uses_only_one_model_call_and_keeps_semantics_pending(monkeypatch, tmp_path):
    import openai

    prepared, response = fixture_response()
    source = tmp_path / "prepared.json"
    source.write_text(prepared.model_dump_json(), encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-unit-test-key")
    dotenv = Mock()
    monkeypatch.setattr(nim, "load_dotenv", dotenv)
    monkeypatch.setattr(nim, "INPUT_PATH", source)
    monkeypatch.setattr(nim, "OUTPUT_PATH", output)
    client = Mock()
    client.chat.completions.create.side_effect = [
        FakeStream([chunk(json.dumps(response)), chunk(finish="stop")]),
        AssertionError("The default runner must not make a second model call"),
    ]
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(openai, "OpenAI", constructor)
    nim.main()
    dotenv.assert_called_once_with(nim.PROJECT_ROOT / ".env")
    constructor.assert_called_once_with(
        base_url=nim.BASE_URL, api_key="fake-unit-test-key", max_retries=0, timeout=1200,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "references_validated"
    assert result["structured_output"]["claims"][0]["evidence_ids"] == response["claims"][0]["evidence_ids"]
    assert "fake-unit-test-key" not in output.read_text(encoding="utf-8")
    assert client.chat.completions.create.call_count == 1
    assert not output.with_suffix(".semantic.json").exists()
    verification = json.loads(output.with_suffix(".verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "needs_review"
    assert verification["review_method"] == "source_comparison"
    assert verification["source_comparison_pending"] is True


def test_main_fails_when_source_checks_fail_despite_valid_response_refs(monkeypatch, tmp_path):
    import openai

    prepared, response = fixture_response()
    prepared.source_records[0].raw_record["targetFromSourceId"] = "WRONG_TARGET"
    source = tmp_path / "prepared.json"
    source.write_text(prepared.model_dump_json(), encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-unit-test-key")
    monkeypatch.setattr(nim, "load_dotenv", Mock())
    monkeypatch.setattr(nim, "INPUT_PATH", source)
    monkeypatch.setattr(nim, "OUTPUT_PATH", output)
    client = Mock()
    client.chat.completions.create.return_value = FakeStream([
        chunk(json.dumps(response)), chunk(finish="stop"),
    ])
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(openai, "OpenAI", constructor)
    with pytest.raises(SystemExit) as stopped:
        nim.main()
    assert stopped.value.code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "references_validated"
    assert json.loads(output.with_suffix(".verification.json").read_text(encoding="utf-8"))["status"] == "failed"
