# KRAS contract-fix result — 2026-09-05

## Changes under test

1. `QueryManifestV2.generated_at` is optional to the model and defaulted by
   Python; Python still overwrites it with authoritative runtime time.
2. The prompt shows the exact fixed `characteristics` shape and allowed values.
3. The prompt says `entity_refs` may use only top-level declared entity refs;
   step outputs belong to data flow and are not entity refs.

## Result

All 77 automated tests passed. The real no-example KRAS run still timed out at
1200 seconds (approximately 1290.7 seconds including cleanup). It made three
underlying calls: two completed at 307.41 and 371.22 seconds; the third was
marked timed out at 612.00 seconds. This is fewer than the preceding five-call
diagnostic, but one run cannot establish a reliable reduction.

`generated_at` and `characteristics` no longer caused validation errors. The
remaining repeated error was data/reference integrity: the model continued to
put resolution output names (`resolved_kras_g12c/d`) in `entity_refs`. Its next
candidate also put drug-list output names (`drug_g12c_list`, `drug_g12d_list`)
in `entity_refs`, so correcting only the first pair would not have been enough.
No candidate reached hidden acceptance and no retrieval ran.

## Content observations

The later candidate improved by adding explicit `retrieve_mechanisms` steps,
but still contained quality problems independent of serialization:

- it classified KRAS G12C/G12D as `mechanism` entities rather than targets;
- it added clinical-development evidence not requested by the question;
- it claimed completion should mean all drug lists and mechanism descriptions
  are complete, an unrealistic completeness condition for bounded databases;
- it selected comparison/ranking metadata without planning those analyses.

## Decision needed before another edit

The prompt-only entity-reference fix failed. Two defensible designs remain:

1. Keep `entity_refs` strict and let Python normalize a name only when it is an
   output of an explicitly depended-on prior step; record every correction and
   continue to reject all other unknown references.
2. Add an explicit `input_refs` field for step-output data and reserve
   `entity_refs` for top-level research entities. This is semantically cleaner
   but changes the manifest contract and requires broader migration/tests.

The first is the smaller experiment. It changes only representation, not which
research operations exist. Neither option should silently fix the incorrect
entity type or research-content omissions.
