# KRAS safe reference-normalization result — 2026-09-06

## Outcome

The real no-example KRAS run completed in one underlying model call (289.63 s)
without timeout. The Plan passed `QueryManifestV2` schema validation and reached
hidden semantic acceptance for the first time in these KRAS diagnostics. No
retrieval ran.

Python removed four misplaced direct-dependency output names from `entity_refs`:
`resolved_kras_g12c`, `resolved_kras_g12d`, `drug_g12c_list`, and
`drug_g12d_list`. Each correction is present in `corrected_model_claims`.
The declared `depends_on` edges remained unchanged. No arbitrary or indirect
unknown reference was accepted, and no research operation was inserted.

## Remaining research-quality failures

The Plan failed hidden acceptance because:

- both KRAS variants were typed as `mechanism`, not `target`;
- the required safeguards `missing_not_negative`,
  `provenance_pagination_truncation`, and `variant_specificity_required` were
  absent;
- the explicit `target_variant` filter was absent;
- the `group_by_variant` aggregation was absent.

Its operations and dependencies were otherwise accepted: variant resolution,
target-drug retrieval and mechanism retrieval were all present. It also used
unsupported completeness language (`all` drugs/mechanisms complete) and added
clinical-development evidence beyond the question.

## Interpretation

Safe normalization solved the retry/serialization blocker in this run. It did
not make the scientific Plan pass, and does not conceal that failure. The next
work is now legitimately Plan-quality evaluation (including the paused
few-shot comparison), rather than further debugging this reference format.
