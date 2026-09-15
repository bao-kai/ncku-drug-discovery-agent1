# KRAS Planner underlying-call diagnosis — 2026-09-05

## Confirmed cause

One outer Planner attempt produced five underlying model calls. This was not
hidden acceptance repair: no candidate Plan reached the acceptance evaluator.
LangChain's structured-output tool rejected each candidate and returned the
validation message to the model, growing the conversation each time.

| Call | Duration | Result / next validation error |
| --- | ---: | --- |
| 1 | 269.66 s | Missing `characteristics` and `generated_at` |
| 2 | 190.81 s | Added `characteristics` as arbitrary boolean flags; missing required `primary` |
| 3 | 291.81 s | Nested the arbitrary flags under `primary`; `primary` must be one allowed string |
| 4 | 228.98 s | Corrected `primary`, but used resolution output names as undeclared `entity_refs` |
| 5 | 309.54 s at outer deadline | Timed out before returning; no raw response |

Total outer generation deadline was 1200 seconds. Including cleanup, the run
ended at approximately 1290.8 seconds. Ollama stopping did not finish within
the 90-second grace period. No evidence retrieval or hidden-rubric scoring ran.

## Interpretation

The repeated requests are confirmed schema/tool correction retries. The model
does not reliably understand two contract details from the generated schema:

1. `characteristics` is descriptive metadata with a fixed shape: `primary`
   is one enum string and `secondary` is a list, not arbitrary boolean traits.
2. `entity_refs` may contain only declared entity reference IDs. Resolution
   outputs belong in data flow; output names are not new entity references.

`generated_at` is runtime metadata that Python overwrites after validation, yet
the model is currently required to generate it. This is an avoidable failure.

The latest rejected candidate also had research-content problems: it combined
drug and mechanism retrieval instead of explicitly using `retrieve_mechanisms`,
introduced an unrequested inhibitor-only filter and minimum-three-drugs rule,
and requested chemical structure although the question did not.

## Recommended minimal experiment

Do not add another Planner stage. Default runtime-owned `generated_at` in
Python, show the exact `characteristics` shape, and clarify that `entity_refs`
can only reuse declared refs. Then rerun this same no-example KRAS diagnostic
once before resuming the few-shot comparison.

The raw trace is in `kras_variant_drugs_r1_off.json`. It contains complete user
questions and prompts; review it before external sharing.
