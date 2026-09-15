# Model Configuration and Replacement

All model selection is centralized in `model_factory.py`. Planner modules call `get_chat_model()`; do not hard-code a model inside individual planners.

## Option A — Replace the Ollama model without code changes

1. Install the model:

```powershell
ollama pull <model-tag>
```

2. Edit the local `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_PROFILE=development
OLLAMA_MODEL=<model-tag>
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=4096
LLM_TEMPERATURE=0
OLLAMA_KEEP_ALIVE=30m
```

`OLLAMA_MODEL` overrides the model selected by `OLLAMA_PROFILE`. The profile name must still be one of `lightweight`, `development`, or `validation`.

3. Verify Ollama:

```powershell
ollama list
ollama run <model-tag> "Return a JSON object with one key named ok."
```

4. Run model-factory tests, then one plan:

```powershell
python -m pytest -q tests/test_model_factory.py -p no:cacheprovider
python agent.py plan-query --max-attempts 1 --question "Your test question" --output outputs/plans/model_smoke_test.json
```

Changing a tag does not prove compatibility. Review the sibling report for schema validity, model calls, timeout, and semantic gaps.

## Option B — Use a remote Ollama server

Keep `LLM_PROVIDER=ollama` and change:

```dotenv
OLLAMA_BASE_URL=http://<server-host>:11434
OLLAMA_MODEL=<server-model-tag>
```

The remote server must already have the model. Protect the endpoint at the network layer; this project does not add authentication to Ollama.

## Option C — Use the supported Anthropic backend

Install the optional dependency:

```powershell
pip install -r requirements-anthropic.txt
```

Configure the local `.env`:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=<local-secret>
ANTHROPIC_MODEL=<supported-model-id>
LLM_TEMPERATURE=0
```

Never put a real key in `.env.example` or commit `.env`.

Ollama-specific timeout cleanup and keep-alive behavior do not apply to Anthropic.

## Option D — Add another provider

Other providers are not currently supported by configuration alone. A contributor must:

1. Add the provider dependency to a separate optional requirements file.
2. Add a new explicit provider branch in `get_chat_model()`.
3. Read the model ID and credentials from environment variables.
4. Return a LangChain-compatible chat model.
5. Add tests for selection, missing credentials, invalid provider, and parameter mapping.
6. Verify both direct one-shot JSON generation and any staged Planner path.
7. Record provider, exact model ID/version, context limit, output limit, and timeout behavior in diagnostics.
8. Update `.env.example` using placeholders only.

Do not give retrieval tools to the Natural-language Planner when adding a provider.

## Minimum practical model capabilities

A replacement model should:

- Follow Traditional Chinese and English technical instructions.
- Return one complete JSON object after optional thinking text.
- Handle the full operation catalog and schema within its context window.
- Preserve exact snake_case identifiers.
- Distinguish `step_id`, `depends_on`, `outputs`, and `entity_refs`.
- Avoid inventing verified biomedical IDs.
- Support stable low-temperature generation.
- Complete within configured attempt and total deadlines.

## Context and output sizing

`OLLAMA_NUM_CTX` is the input/context capacity. `OLLAMA_NUM_PREDICT` bounds reasoning plus final output. Too little context may omit operation contracts; an excessive context or output budget can cause slow context shifting or memory pressure.

Start with:

```dotenv
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=4096
```

Then measure rather than guessing. Smaller hardware may require a smaller model or context.

## Model replacement acceptance checklist

- Offline tests pass.
- The model is reported under the intended provider and ID.
- One simple plan is schema-valid.
- One representative complex development case is evaluated.
- No retrieval runs during plan-only tests.
- No unverified IDs are accepted.
- Runtime, attempt count, timeout, and model digest/version are recorded.
- Scientific quality is manually reviewed; passing schema is not enough.