# Few-shot evaluation after contract fixes — 2026-09-06

## Controlled results

All runs used qwen3:14b, temperature 0, full operation table, 16384 context,
one outer Planner attempt, no hidden-rubric repair and no retrieval. All six
Plans completed in one underlying model call and passed schema validation.

| Case | Mode | Repetitions | Acceptance passes | Mean time |
| --- | --- | ---: | ---: | ---: |
| KRAS variant drugs | no examples | 2 | 0/2 | 298.29 s |
| KRAS variant drugs | synthetic_v1 | 2 | 0/2 | 334.37 s |
| PDAC known targets | no examples | 1 | 0/1 | 392.43 s |
| PDAC known targets | synthetic_v1 | 1 | 0/1 | 311.87 s |

Timing directions disagree by case. These samples establish neither a speed
benefit nor a penalty. The former schema-retry blocker no longer contaminates
this comparison.

## KRAS findings

Across both repetitions, few-shot consistently typed both variants as targets,
kept the three necessary capabilities, included provenance and missing-not-
negative safeguards, and needed no reference normalization. Both runs still
omitted `variant_specificity_required`, the explicit `target_variant` filter
and `group_by_variant`. Baselines had 6-7 rubric gaps versus 3 in each few-shot
run. One few-shot run added unrequested aggregation/ranking; the other did not.

## PDAC findings

The baseline contained only disease resolution and target discovery. Few-shot
added stratification and ranking, plus some safeguards, but did not retrieve
the drug or tractability inputs those analyses require. It placed ChEMBL and
ClinicalTrials.gov on an analysis step instead of explicit retrieval steps,
used the wrong ranking metric, invented a `therapeutic_target` filter, and split
one bilingual disease mention into two disease entities. It still failed.

## Conclusion

`synthetic_v1` shows repeatable partial improvement on the KRAS development
case, but no acceptance pass and no reliable cross-question quality. Keep
`PLANNER_FEW_SHOT_MODE=off`. Before another version, review example coverage:
neither current example demonstrates variant-specific target-to-drug-to-
mechanism planning, while PDAC shows that adding stratify/rank labels without
upstream evidence is not enough. Any development case structurally mirrored by
a new example cannot count as unseen-generalization evidence.
