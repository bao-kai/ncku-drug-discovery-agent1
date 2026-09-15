# Few-shot comparison: partial results, 2026-09-04

## Scope and status

The batch ended through its configured timeout guard: 3 of 8 planned runs
were attempted. Two produced Plans; the third timed out. Five runs were not
started. This is one matched pair only, not a stability or generalization result.
No evidence retrieval took place. The production few-shot default stays off.

Both modes used qwen3:14b, temperature 0, the full operation table and 16384
context tokens. The earlier context-4096 aborted run is excluded.
No hidden-rubric repair was supplied. However, the structured-output agent
can issue multiple underlying model calls within one Planner attempt. Server
logs showed repeated requests for the KRAS baseline; exact retry causes are
not established by those logs. Thus these are first Planner attempts, not
necessarily single model calls.

| Run | Time | Result |
| --- | ---: | --- |
| PDAC known targets, no examples, round 1 | 698.02 s | Failed acceptance |
| PDAC known targets, synthetic_v1, round 1 | 674.35 s | Failed acceptance |
| KRAS variant drugs, no examples, round 1 | 1290.58 s including cleanup | Generation timeout at 1200 s; Stopping timeout after grace period |

## Semantic review of the matched pair

The baseline planned disease resolution and associated-target discovery only.
It omitted target-drug retrieval, tractability, evidence stratification and
ranking, along with required safeguard/ranking/aggregation semantics.

The few-shot Plan added evidence stratification and several safeguards. It
still failed to retrieve the drug and tractability evidence needed by that
stratification, omitted final ranking and the required aggregation definition,
and placed source names on an analysis step without actual retrieval steps.
It also introduced an unrequested association-score threshold >0.3 and
drug/tractability filters that could exclude candidates without authorization.
The generic phrase 'therapeutic targets' was incorrectly extracted as a named
target entity. These issues matter beyond whether canonical operation names
appear in the output.

The examples' literal synthetic disease names and >0.4 threshold were not
copied. The invented >0.3 threshold is a scope error, but this single pair
cannot establish that few-shot caused it.

## Conclusion and limits

There is partial improvement in explicit stratification/safeguard wording, not
evidence of a sufficiently complete or dependable Plan. Do not enable the
examples by default based on this result. The approximately 24-second latency
difference is not meaningful evidence of a speed improvement with one pair.

The second task has no matched few-shot result; repetition and cross-task
claims remain untested. The next controlled comparison requires the model to
be confirmed idle and the repeated structured-output requests to be diagnosed.
Do not restart requests while model stopping is incomplete. No automatic
server kill or restart was performed.

Post-batch verification: Ollama subsequently returned an empty running-model
list. This confirms later unloading, not successful cleanup within the grace
period. The batch remains terminated and was not automatically restarted.
