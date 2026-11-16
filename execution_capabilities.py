"""Shared executable contracts for runtime planning and deterministic execution."""
import json
from target_evidence_inventory import validate_numeric_filters
from planner_acceptance import OPERATION_TABLE

RESOLVE = {'resolve_disease', 'resolve_diseases', 'resolve_target', 'resolve_targets',
           'resolve_disease_target', 'resolve_disease_targets'}
PAIR = {'retrieve_exact_pair', 'retrieve_exact_pairs', 'retrieve_datasource_scores'}
TRIAL = {'retrieve_clinical_trials', 'retrieve_clinical_stage',
         'retrieve_trial_status_and_stop_reasons', 'retrieve_trial_terminations'}
LITERATURE = {'retrieve_negative_publications', 'retrieve_safety_evidence', 'retrieve_publications'}
SOURCES = {op: {'open_targets'} for op in RESOLVE | PAIR |
           {'discover_targets', 'discover_targets_per_disease', 'retrieve_target_drugs', 'retrieve_tractability'}}
SOURCES.update({op: {'clinicaltrials_gov'} for op in TRIAL})
SOURCES.update({op: {'europe_pmc'} for op in LITERATURE})
SOURCES['retrieve_safety_evidence'].add('clinicaltrials_gov')
SOURCES['retrieve_target_drugs'].add('chembl')
SOURCES['rank_targets'] = set()
SOURCES['stratify_target_evidence_layers'] = set()
SOURCES['filter_targets'] = set()
SOURCES['retrieve_normal_tissue_expression'] = {'open_targets'}
METRICS = {'association_score', 'overall_association_score', 'known_drug_count'}
INVENTORY_AGGREGATION = {'target_translation_layer': {'method': 'source_evidence_inventory_v1'}}


def validate_runtime_repair(base, candidate):
    """An unavailable capability remains unresolved, not repaired by dropping it.

    Registered operation aliases are normalized before this check. No other
    equivalence between an unavailable operation and an executable one is defined.
    This is a repair regression check, not a semantic evaluator of the first Plan.
    """
    candidates = {s.step_id: s for s in candidate.plan_steps}
    for step in base.plan_steps:
        if step.operation in SOURCES:
            continue
        replacement = candidates.get(step.step_id)
        if replacement is None or (
            replacement.operation_status == 'approved' and replacement.operation in SOURCES
        ):
            operation = step.proposed_operation_name or step.operation
            raise ValueError(
                f'step_id={step.step_id}; unavailable operation {operation} cannot be '
                'removed or replaced with an executable operation without a registered '
                'equivalent binding. Preserve the unresolved capability; do not weaken '
                'the research plan to pass runtime validation.'
            )


def runtime_findings(plan, human_confirmed_context=None):
    findings = []
    ancestors = {}
    by_id = {s.step_id: s for s in plan.plan_steps}
    entity_types = {entity.entity_ref: entity.entity_type for entity in plan.entities}
    for s in plan.plan_steps:
        ancestors[s.step_id] = set(s.depends_on)
        for dep in s.depends_on:
            ancestors[s.step_id].update(ancestors[dep])
        def issue(message,missing=None,input_entity_refs=None):
            item={'step_id': s.step_id, 'operation': s.operation, 'problem': message,
                  'required_correction': 'Use the executable operation/source/method/metric contract supplied in the prompt; do not omit necessary research requirements.'}
            if missing: item['missing']=missing
            if input_entity_refs: item['input_entity_refs']=input_entity_refs
            findings.append(item)
        allowed = SOURCES.get(s.operation)
        if s.operation_status != 'approved' or allowed is None:
            issue(f'operation not implemented: {s.operation}')
            continue
        if set(s.sources) - allowed or (allowed and not s.sources):
            issue(f'executable sources are {sorted(allowed)}; got {s.sources}')
        if len(s.sources) != len(set(s.sources)):
            issue('duplicate sources')
        if s.operation in PAIR:
            disease_refs = [ref for ref in s.entity_refs if entity_types[ref] == 'disease']
            target_refs = [ref for ref in s.entity_refs if entity_types[ref] == 'target']
            if len(disease_refs) != 1 or not target_refs or (
                s.operation == 'retrieve_exact_pair' and len(target_refs) != 1
            ):
                issue('Exact pair binding requires one disease and explicit target(s); retrieve_exact_pair requires exactly one target')
            for kind, refs in [('disease', disease_refs), ('target', target_refs)]:
                capability = f'resolved_{kind}'
                bound = {
                    ref for dep in ancestors[s.step_id]
                    if by_id[dep].operation in RESOLVE
                    and by_id[dep].operation_status == 'approved'
                    and capability in OPERATION_TABLE[by_id[dep].operation].provides
                    for ref in by_id[dep].entity_refs
                }
                missing_refs = [ref for ref in refs if ref not in bound]
                if missing_refs:
                    issue(f'Exact pair requires upstream {capability} for {missing_refs}',
                          [capability], missing_refs)
        if s.operation in TRIAL | LITERATURE and not s.entity_refs:
            issue('Search requires explicit entity_refs')
        uses_drug_count = False
        if s.operation == 'filter_targets':
            try:
                validate_numeric_filters(s.filters)
            except ValueError as exc:
                issue(str(exc))
            else:
                uses_drug_count = any(predicate['field'] == 'known_drug_count'
                                      for predicate in s.filters['all'])
        elif s.filters:
            issue('filters have no executable binding')
        if s.operation == 'stratify_target_evidence_layers':
            if s.aggregation != INVENTORY_AGGREGATION:
                issue('Use exactly aggregation=' + json.dumps(INVENTORY_AGGREGATION) + '; this names a fully defined native inventory algorithm, not free-text interpretation.')
        elif s.aggregation:
            issue('aggregation has no executable binding for this operation')
        if s.ranking:
            metrics = {s.ranking.primary_metric, *s.ranking.secondary_metrics}
            uses_drug_count = uses_drug_count or 'known_drug_count' in metrics
            expression_ranking=(s.operation=='retrieve_normal_tissue_expression' and
                                s.ranking.primary_metric=='expression_median' and not s.ranking.secondary_metrics)
            if not expression_ranking and (s.operation != 'rank_targets' or metrics - METRICS):
                issue(f'executable ranking metrics: {sorted(METRICS)}; no tractability_score is defined')
        if uses_drug_count and not any(
            by_id[d].operation == 'retrieve_target_drugs' and 'open_targets' in by_id[d].sources
            for d in ancestors[s.step_id]
        ):
            issue('known_drug_count requires upstream retrieve_target_drugs from open_targets')
        if s.operation == 'rank_targets' and not s.ranking:
            issue('rank_targets requires ranking')
        if s.operation in RESOLVE and not s.entity_refs:
            issue('resolution requires entity_refs')
        if human_confirmed_context is not None and s.operation not in RESOLVE:
            for entity in plan.entities:
                if entity.entity_ref not in s.entity_refs or human_confirmed_context.binding_for(entity) is None:
                    continue
                if not any(by_id[d].operation in RESOLVE and entity.entity_ref in by_id[d].entity_refs for d in ancestors[s.step_id]):
                    capabilities=sorted({cap for op in RESOLVE
                        if set(OPERATION_TABLE[op].allowed_entity_types)=={entity.entity_type}
                        for cap in OPERATION_TABLE[op].provides})
                    issue(f'Human-confirmed entity {entity.entity_ref} requires upstream native resolution; scope confirmation is not source identity proof',
                          capabilities,[entity.entity_ref])
    return findings


def runtime_contract_prompt():
    return (
        'Executable source bindings (authoritative for this run):\n' +
        json.dumps({op: sorted(sources) for op, sources in sorted(SOURCES.items())}) +
        '\nExact-pair and datasource-score steps require one explicit disease ref and explicit target refs, '
        'each verified by a resolver in that step\'s ancestors. retrieve_exact_pair requires one target; '
        'retrieve_exact_pairs and retrieve_datasource_scores allow multiple targets and return overall and datasource scores. '
        'Discovery candidates do not replace explicit exact-pair target bindings. '
        'Each trial or literature step uses its entity_refs as jointly required search constraints (AND). '
        'A disease and target can constrain one scope; use separate steps for independent subjects. '
        'Description and completion_criteria do not generate keywords, status filters, or limits. ' +
        '\nOnly filter_targets accepts filters. Use an all array of field/operator/value objects, '
        'with the requested numeric value and registered strict operator gt. '
        'Numeric fields are overall_association_score, association_score and known_drug_count. Predicates in all are AND. '
        'Additionally {"field":"predicted_small_molecule_pocket","operator":"is_true","value":true} requires '
        'upstream retrieve_tractability and selects native SM High-Quality Pocket OR Med-Quality Pocket true. '
        'Both false means false; missing/null without a true flag stays unknown. Approved Drug and other generic flags do not qualify. '
        'To INCLUDE predictions alongside experimental source evidence, use '
        '{"field":"small_molecule_pocket_evidence","operator":"is_true","value":true}: '
        'native SM High-Quality Pocket OR Med-Quality Pocket OR Structure with Ligand true, requiring upstream retrieve_tractability. '
        'This inclusive eligibility flag preserves predicted_pocket_flags and experimental_structure_with_ligand separately; '
        'it never relabels computational predictions as experimentally verified pockets or coordinate-level pocket geometry. '
        'predicted_small_molecule_pocket alone intentionally excludes experimental-only candidates and is only suitable when predictions alone are requested. '
        'Keep experimental_structure_with_ligand (native SM Structure with Ligand co-crystal flag) separate from predictions. '
        'gt is strict; unknown stays unknown, never zero. This filters only retrieved candidates, not the whole database. '
        'No physical-pocket predicate is registered: predicted tractability pockets cannot be silently treated as experimentally known sites. '
        'All other filters must be {}. rank_targets consumes retrieved target rows, including exact-pair results, '
        'and accepts ranking with metrics '
        'association_score (alias of overall_association_score), overall_association_score and known_drug_count. '
        'known_drug_count is the count of distinct native Open Targets drug.id values in the retrieved target drug rows, '
        'across all reported indications, not disease-specific approval. Truncated results are observed lower bounds. '
        'For both filtering and ranking this metric requires upstream retrieve_target_drugs with open_targets. Missing data stays unknown, not zero. '
        'No scalar tractability_score is defined; retain native label/modality/value without scoring it. '
        '\nFor stratify_target_evidence_layers use EXACTLY aggregation=' + json.dumps(INVENTORY_AGGREGATION) + '. '
        'This registered method preserves every disease-target candidate, left-joins source records by native target ID, '
        'keeps association, drug and tractability as separate overlapping descriptive dimensions, '
        'preserves original drug indications and source pointers, and retains missing data as unknown. '
        'ChEMBL protein-component mappings remain separately labelled context, not confirmed single-gene drug binding. '
        'No efficacy/approval/support tier or clinical strength is inferred. No extra aggregation keys are executable. '
        'Other operations use aggregation=null. Analysis steps have sources=[]. '
        'Discovery retrieves at most the existing first 10 candidates per disease. '
        'ClinicalTrials.gov searches return the first page of up to 20 native trial records with truncation metadata. '
        'Only retrieve_trial_terminations applies the TERMINATED status filter. '
        'Trial phase is a native study field, not a computed drug-program phase. '
        'Europe PMC safety and negative-literature operations add their registered topic terms. '
        'ClinicalTrials.gov safety retrieval searches the entity constraints without a safety-specific topic filter. '
        'All hits are candidates for review. '
        'ChEMBL target-drug retrieval first uses Open Targets for native protein accessions, then retrieves bounded ChEMBL mapping and mechanism pages. '
        'retrieve_publications searches the referenced literal topics and confirmed disease scope without requiring canonical target IDs, '
        'up to 20 relevance-ranked records per query and no implicit date cutoff. Do not require a parent-gene ID just to search an isoform topic. '
        'Search hits do not resolve isoform identity; any gene association is gene-level background only. '
        'Evidence content (mutation/expression, possibly overlapping) differs from provenance (literature/database). '
        'The existing downstream organizer supports source-linked content_annotations with explicit model-interpretation labels; '
        'do not invent an executor aggregation just to inventory these categories. Unclassified material stays unknown. '
        'Scores, article counts and clinical stages are not counts/strength of evidence or a universal main-evidence conclusion. '
        'retrieve_normal_tissue_expression returns all available native baseline expression pages per target (no first-20 cap) with native assay/datasource/unit, '
        'and optionally accepts ranking with primary_metric="expression_median", secondary_metrics=[], direction="descending" '
        'to present relatively higher source medians within each native target/source/datatype/unit/cell-type/quality-control scope. '
        'Native tissue/source order breaks equal-median ties; missing or uninterpretable medians remain unknown after known rows. '
        'No averaging, pooled RNA/protein, absolute high cutoff or clinical safety inference is performed. '
        'High specificity is not high abundance. '
        'If the question requires an unsupported operation, do not substitute a weaker research plan to pass validation.'
    )
