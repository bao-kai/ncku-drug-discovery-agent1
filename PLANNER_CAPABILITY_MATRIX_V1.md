# Planner Capability Matrix v1

This matrix is the reviewed phase-1 Plan-only acceptance map. The six query
characteristics remain descriptive metadata; allocation is based on research
capabilities, paths, and evidence constraints.

## Reviewed decision rules

1. "Therapeutic targets" must be presented in two layers: disease-associated
   targets, then targets with drug evidence or tractability evidence.
2. Existing databases may provide preliminary approval records, but official
   regulatory-source confirmation must be requested before a definitive claim.
3. Shared targets report scores separately per disease; no synthetic combined
   ranking is required.
4. A no-drug statement must be bounded by searched sources and retrieval date.
5. Mechanism distributions are classified and counted before categories are
   ranked by count.
6. Current normal-tissue expression evidence is preliminary; formal safety use
   requires confirmation from a specialized expression source.
7. Ranking is mandatory when the question explicitly asks for strongest, most,
   priority, or rank. For large result sets it is optional only when the Plan
   states a defensible ranking basis.

## Capability matrix

`D` is the development set and `H` is the internal holdout set. `F`, `R`, `A`,
and `N` mean filtering, ranking, aggregation/comparison, and negative/safety
knowledge respectively.

| # | Set | Case | Core path | F | R | A | N | Sources |
|---:|:---:|---|---|:---:|:---:|:---:|:---:|---|
| 1 | D | pdac_known_targets | disease -> targets -> drug/tractability evidence -> two layers -> rank |  | Y | Y |  | OT, ChEMBL |
| 2 | H | kras_associated_diseases | target -> diseases -> rank -> PDAC relative position |  | Y | Y |  | OT |
| 3 | D | kras_variant_drugs | variants -> drugs -> mechanisms, grouped by variant | Y |  | Y |  | OT, ChEMBL |
| 4 | D | olaparib_targets_indications | drug -> targets/approval records -> PDAC check | Y |  |  | Y | OT, ChEMBL |
| 5 | H | pdac_drug_repurposing | disease -> targets -> drugs -> join -> rank | Y | Y | Y |  | OT, ChEMBL |
| 6 | D | pdac_drugs_other_cancers | disease/drugs -> targets -> other cancers | Y |  | Y |  | OT, ChEMBL |
| 7 | D | pdac_nsclc_shared_targets | diseases -> targets -> intersection -> per-disease scores |  |  | Y |  | OT |
| 8 | D | cldn182_pdac_evidence_types | exact pair -> datasource evidence -> evidence-type comparison | Y | Y | Y | Y | OT, Europe PMC |
| 9 | H | kras_pdac_datasource_contribution | exact pair -> datasource scores -> contribution rank |  | Y | Y |  | OT |
| 10 | D | pdac_genetic_supported_without_clinical_drug | targets -> genetic evidence/drugs/stage -> source-bounded filter | Y | Y | Y | Y | OT, CT.gov, ChEMBL |
| 11 | D | pdac_score_tractability_filter | targets -> tractability -> threshold filter -> rank | Y | Y |  |  | OT |
| 12 | H | pdac_phase2_not_approved | targets -> drugs -> trials/indications -> filter | Y | Y | Y | Y | OT, CT.gov, ChEMBL |
| 13 | D | pdac_first_in_class_opportunity | targets -> evidence/drugs/tractability -> filter -> rank | Y | Y | Y | Y | OT, ChEMBL |
| 14 | D | compare_kras_cldn182_pdac | exact pairs -> evidence types -> normalize -> compare | Y |  | Y | Y | OT, Europe PMC |
| 15 | D | pdac_clinical_mechanism_distribution | trials -> interventions -> classify -> count -> rank | Y | Y | Y | Y | CT.gov, ChEMBL, OT |
| 16 | H | kinase_drug_coverage_pdac_vs_lung | disease targets -> kinases -> drugs -> denominators -> rates | Y |  | Y | Y | OT, ChEMBL |
| 17 | H | shp2_fak_safety_expression | safety/terminations/expression -> integrate, require specialist confirmation | Y | Y | Y | Y | OT, Europe PMC, CT.gov |
| 18 | D | pdac_withdrawn_terminated_drugs | targets -> drugs -> status/literature -> reasons/program aggregation | Y | Y | Y | Y | all four |

## Allocation

- Development (12): 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 18.
- Internal holdout (6): 2, 5, 9, 12, 16, 17.

The holdout is not a true unseen external test because all 18 questions were
already reviewed during development. It is a change-control set: do not tune a
prompt or evaluator to one holdout failure. A future expert-authored unseen set
is still required for a strong generalization claim.
