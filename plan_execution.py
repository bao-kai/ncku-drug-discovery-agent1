"""Deterministic QueryManifestV2 -> source tools. No model or prose execution."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from uuid import uuid4

import query_opentargets as ot
import query_chembl as chembl
import requests
from query_clinicaltrials import search_clinical_trials
from query_europepmc import search_publications
from operation_rules import evaluate_operation_rules, evaluate_question_requirements
from schemas import QueryManifestV2
from human_entity_context import HumanConfirmedEntityContext, HumanResearchClarification
from execution_capabilities import RESOLVE, PAIR, TRIAL, LITERATURE, SOURCES, METRICS, runtime_findings
from target_evidence_inventory import target_inventory, rank_inventory, filter_inventory, rank_expression_material


def now():
    return datetime.now(timezone.utc).isoformat()


def execution_issues(plan, human_confirmed_context=None, research_clarification=None):
    """Reject non-executable instructions before spending any API calls."""
    requirements = evaluate_question_requirements(plan, research_clarification, human_confirmed_context)
    return _execution_issues(plan, human_confirmed_context, requirements)


def _execution_issues(plan, human_confirmed_context, requirement_findings):
    issues = [f'{f.step_id or "plan"}: {f.problem}'
              for f in evaluate_operation_rules(plan).blocking_findings]
    issues.extend(f"{f['step_id']}: {f['problem']}" for f in runtime_findings(plan,human_confirmed_context))
    issues.extend(f'{f.step_id or "plan"}: {f.problem}' for f in requirement_findings if f.severity != 'warning')
    return issues


class PlanExecutor:
    def __init__(self):
        self.report = None
        self.human_confirmed_context = None

    def _call(self, step, source, tool, args, fn):
        call = {'call_id': f'{step.step_id}:call_{len(self.report["calls"]) + 1}',
                'step_id': step.step_id, 'source': source, 'tool': tool,
                'arguments': deepcopy(args), 'started_at': now(), 'status': 'running'}
        self.report['calls'].append(call)
        if source == 'open_targets':
            call['source_url'] = ot.ENDPOINT
        elif source == 'chembl':
            call['source_url'] = requests.Request('GET', f'{chembl.CHEMBL_ENDPOINT}/{args["path"]}',
                                                  params=args['params']).prepare().url
        try:
            raw = fn(**args)
            call['raw_response'] = raw
            if isinstance(raw, dict) and raw.get('source_url'):
                call['source_url'] = raw['source_url']
            if isinstance(raw, dict) and raw.get('status') == 'error':
                raise RuntimeError(raw.get('error') or 'Source returned error')
            call['status'] = 'success'
            return raw, call['call_id']
        except Exception as exc:
            call.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            call['finished_at'] = now()

    def _graphql(self, step, query, variables):
        raw, ref = self._call(step, 'open_targets', 'graphql',
                              {'query': query, 'variables': variables}, ot._graphql)
        self.report['calls'][-1]['source_url'] = ot.ENDPOINT
        return raw, ref

    def _resolve(self, s, entities):
        result = []
        for e in entities:
            kind = e.entity_type
            if kind not in {'disease', 'target'}:
                raise ValueError(f'No resolver for {kind}')
            if e.search_candidates and not e.entity_id:
                result.append(self._resolve_search_candidates(s, e))
                continue
            entity_id = e.entity_id
            details = {}
            proof = None
            if not entity_id:
                # Literal parenthetical aliases only; do not infer synonyms or
                # fall back from a specified subtype to its broader disease.
                aliases = [v.strip() for part in re.findall(r'[（(]([^）)]+)[）)]', e.mention)
                           for v in re.split(r'[,，;；]', part) if v.strip()]
                queries = list(dict.fromkeys(sorted(aliases, key=len, reverse=True) + [e.canonical_name or e.mention]))
                candidates, exact = [], []
                names = {ot._normalized_entity_name(q) for q in queries}
                for query_name in queries:
                    raw, _ = self._graphql(s, ot.SEARCH_QUERY, {'queryString': query_name, 'entityNames': [kind]})
                    candidates = list({c['id']:c for c in [*candidates, *raw['search']['hits']]}.values())
                    exact = [c for c in candidates if ot._normalized_entity_name(c['name']) in names]
                    if exact:
                        break
                def fetch_identity(candidate_id):
                    query = ot.DISEASE_ALIAS_QUERY if kind == 'disease' else ot.TARGET_ALIAS_QUERY
                    key = 'efoId' if kind == 'disease' else 'ensemblId'
                    native, call_ref = self._graphql(s, query, {key: candidate_id})
                    details[candidate_id] = (native.get(kind), call_ref)
                    return native.get(kind)
                chosen, proof, _ = ot.choose_source_candidate(queries, kind, candidates, fetch_identity)
                entity_id = chosen['id']
            query = ot.DISEASE_IDENTITY_QUERY if kind == 'disease' else ot.TARGET_IDENTITY_QUERY
            key = 'efoId' if kind == 'disease' else 'ensemblId'
            if entity_id in details:
                row, ref = details[entity_id]
            else:
                raw, ref = self._graphql(s, query, {key: entity_id})
                row = raw.get(kind)
            if not row or row['id'] != entity_id:
                raise ValueError(f'Identity not returned: {entity_id}')
            confirmed = self.human_confirmed_context.binding_for(e) if self.human_confirmed_context else None
            if confirmed is not None:
                native_name = row.get('name') if kind == 'disease' else row.get('approvedSymbol')
                if entity_id != confirmed.entity_id or native_name != confirmed.source_name:
                    raise ValueError('Native source identity contradicts human-confirmed scope ID/name')
            result.append({'entity_ref': e.entity_ref, 'entity_type': kind, 'id': row['id'],
                           'name': row.get('name') or row.get('approvedSymbol'), 'source_ref': ref})
            if proof:
                result[-1]['resolution_evidence'] = dict(**proof, source_ref=ref,
                    candidate_identity_refs=[call_ref for _, call_ref in details.values()])
            if confirmed is not None:
                result[-1]['resolution_evidence'] = dict(
                    human_scope=dict(approval_basis=self.human_confirmed_context.approval_basis,
                        run_id=self.report['run_id'], mention=e.mention, entity_id=confirmed.entity_id,
                        confirmed_mentions=confirmed.mentions,
                        confirmation_record='human_confirmed_context'),
                    source_identity=dict(basis='native_id_and_name', source_ref=ref,
                        id=row['id'], name=native_name),
                    limitation='Human confirms research scope; source confirms native identity, not Chinese synonymy.')
        return {'entities': result}

    def _resolve_search_candidates(self, step, entity):
        """Keep generated search leads separate from native identity and scope."""
        kind = entity.entity_type
        aliases = [v.strip() for part in re.findall(r'[（(]([^）)]+)[）)]', entity.mention)
                   for v in re.split(r'[,，;；]', part) if v.strip()]
        literal_names = [entity.mention, *aliases]
        queries = list(dict.fromkeys([*literal_names, *(c.text for c in entity.search_candidates)]))
        audit = dict(step_id=step.step_id, entity_ref=entity.entity_ref, mention=entity.mention,
                     search_candidates=[c.model_dump() for c in entity.search_candidates],
                     generation_source='planner_unverified_hypothesis', searches=[],
                     identity_status='unresolved', scope_status='unresolved', status='running')
        self.report.setdefault('entity_resolution_attempts', []).append(audit)
        candidates, details = [], {}
        def fetch_identity(candidate_id):
            if candidate_id not in details:
                query = ot.DISEASE_SCOPE_QUERY if kind == 'disease' else ot.TARGET_ALIAS_QUERY
                key = 'efoId' if kind == 'disease' else 'ensemblId'
                raw, ref = self._graphql(step, query, {key: candidate_id})
                details[candidate_id] = (raw.get(kind), ref)
                audit.setdefault('candidate_identity_refs', []).append(ref)
            return details[candidate_id][0]
        try:
            for name in queries:
                raw, ref = self._graphql(step, ot.SEARCH_QUERY, {'queryString': name, 'entityNames': [kind]})
                hits = (raw.get('search') or {}).get('hits') or []
                audit['searches'].append(dict(text=name, source_ref=ref, candidate_ids=[h['id'] for h in hits]))
                candidates = list({c['id']: c for c in [*candidates, *hits]}.values())
            audit['returned_candidates'] = candidates
            if not candidates:
                audit['status'] = 'zero_candidates'
                raise ot.OpenTargetsError('No source candidates for original mention or search leads')
            chosen, proof, _ = ot.choose_source_candidate(queries, kind, candidates, fetch_identity)
            row = fetch_identity(chosen['id'])
            if not row or row.get('id') != chosen['id']:
                raise ot.OpenTargetsError(f'Candidate identity mismatch or unavailable: {chosen["id"]}')
            ref = details[chosen['id']][1]
            query_names = {ot._normalized_entity_name(n) for n in queries}
            native_match = next(((term, pointer, source) for term, pointer, source in ot._identity_alias_terms(row, kind)
                                 if ot._normalized_entity_name(term) in query_names), None)
            if native_match is None:
                raise ot.OpenTargetsError('Candidate identity mismatch: native name/aliases do not match search leads')
            native_term, native_pointer, native_source = native_match
            audit.update(identity_status='verified', candidate_id=row['id'],
                         candidate_selection_proof=proof,
                         candidate_identity_proof=dict(basis='source_identity_name_or_alias',
                             matched_term=native_term, json_pointer=native_pointer, source=native_source, source_ref=ref),
                         source_scope=dict(source_ref=ref, name=row.get('name') or row.get('approvedName'),
                                           description=row.get('description'), parents=row.get('parents'),
                                           ancestors=row.get('ancestors')))
            normalized = {ot._normalized_entity_name(n) for n in literal_names}
            match = next(((term, pointer, source) for term, pointer, source in ot._identity_alias_terms(row, kind)
                          if ot._normalized_entity_name(term) in normalized), None)
            if match is None:
                audit.update(status='needs_review', scope_status='needs_review',
                             scope_limitation='Native English identity/definition/hierarchy does not establish '
                             'equivalence to the original mention; candidate may have a different scope. '
                             'No source-backed original-language exact name/alias was returned.')
                raise ot.OpenTargetsError('Original mention equivalence/scope not source-confirmed; needs_review')
            term, pointer, source = match
            audit.update(status='verified', scope_status='verified', scope_proof=dict(
                basis='source_literal_name_or_exact_alias', matched_term=term,
                json_pointer=pointer, source=source, source_ref=ref))
            return dict(entity_ref=entity.entity_ref, entity_type=kind, id=row['id'],
                        name=row.get('name') or row.get('approvedSymbol'), source_ref=ref,
                        resolution_evidence=dict(identity_proof=audit['candidate_identity_proof'],
                                                 scope_proof=audit['scope_proof']))
        except ot.EntityResolutionError as exc:
            audit.update(status='ambiguous', identity_status='ambiguous', error=f'{type(exc).__name__}: {exc}')
            raise
        except Exception as exc:
            if audit['status'] == 'running':
                audit['status'] = 'identity_mismatch' if 'identity mismatch' in str(exc).lower() else 'source_error'
            audit['error'] = f'{type(exc).__name__}: {exc}'
            raise

    def _execute_step(self, s, entities, upstream):
        if s.operation in RESOLVE:
            return self._resolve(s, entities)
        resolved = {e['entity_ref']: e for data in upstream for e in data.get('entities', [])}
        selected = [resolved[e.entity_ref] for e in entities if e.entity_ref in resolved]
        diseases = [e for e in selected if e['entity_type'] == 'disease']
        targets = [e['id'] for e in selected if e['entity_type'] == 'target']
        # Candidate collections belong to declared ancestors, not global execution state.
        if not targets and s.operation not in PAIR:
            for row in target_inventory(upstream):
                if not diseases or row['disease_id'] in {d['id'] for d in diseases}:
                    targets.append(row['target_id'])
        targets = list(dict.fromkeys(targets))
        if s.operation in ({'discover_targets', 'discover_targets_per_disease',
                            'retrieve_target_drugs', 'retrieve_tractability', 'retrieve_normal_tissue_expression',
                            'filter_targets', 'rank_targets', 'stratify_target_evidence_layers'} | PAIR):
            unresolved = [e.entity_ref for e in entities
                          if e.entity_type in {'disease', 'target'} and e.entity_ref not in resolved]
            if unresolved:
                raise ValueError(f'No verified upstream binding for entities: {unresolved}')
        if s.operation in {'discover_targets', 'discover_targets_per_disease'} | PAIR:
            if not diseases or (s.operation in PAIR and not targets):
                raise ValueError('Missing verified upstream disease/target IDs')
            if s.operation in PAIR and (len(diseases) != 1 or (s.operation == 'retrieve_exact_pair' and len(targets) != 1)):
                raise ValueError('Exact pair binding requires one disease and explicit target(s); no inferred Cartesian product')
            rows, collections = [], []
            for disease in diseases:
                for target in (targets if s.operation in PAIR else [None]):
                    query = ot.DISEASE_TARGET_PAIR_QUERY if target else ot.DISEASE_TARGET_ASSOCIATION_QUERY
                    variables = {'efoId': disease['id']}
                    variables.update({'ensemblId': target} if target else {'pageSize': 10})
                    raw, ref = self._graphql(s, query, variables)
                    record = raw.get('disease')
                    if not record or record['id'] != disease['id']:
                        raise ValueError('Disease identity mismatch')
                    assoc = record['associatedTargets']
                    for index, row in enumerate(assoc['rows']):
                        if target and row['target']['id'] != target:
                            raise ValueError('Exact target identity mismatch')
                        rows.append({'disease_id': disease['id'], 'target_id': row['target']['id'],
                                     'target_symbol': row['target']['approvedSymbol'],
                                     'overall_association_score': row['score'],
                                     'datasource_scores': row['datasourceScores'],
                                     'source_ref': ref, 'pointer': f'/disease/associatedTargets/rows/{index}'})
                    collections.append({'source_ref': ref, 'disease_id': disease['id'],
                                        'total_count': assoc['count'], 'returned_count': len(assoc['rows']),
                                        'truncated': assoc['count'] > len(assoc['rows'])})
            return {'targets': rows, 'collections': collections}
        if s.operation == 'retrieve_normal_tissue_expression':
            if not targets:
                raise ValueError('Missing verified upstream target IDs')
            collections=[]; expression_material=[]
            for target in targets:
                total=None; fetched=0; refs=[]; fingerprints=set()
                while total is None or fetched<total:
                    raw,ref=self._graphql(s,ot.BASELINE_EXPRESSION_QUERY,
                        {'ensemblId':target,'rowOffset':fetched,'pageSize':500})
                    native=raw.get('target')
                    if not native or native.get('id')!=target:
                        raise ValueError('Baseline expression target identity mismatch')
                    data=native['baselineExpression']; count=data['count']; rows=data['rows']
                    if isinstance(count,bool) or not isinstance(count,int) or count<0 or not isinstance(rows,list) or len(rows)>500:
                        raise ValueError('Invalid baseline expression collection')
                    if total is not None and total!=count:
                        raise ValueError('Baseline expression count changed during pagination')
                    total=count
                    if any(not isinstance(r,dict) or r.get('targetId')!=target for r in rows):
                        raise ValueError('Baseline expression row target mismatch')
                    expression_material.extend(dict(record=deepcopy(row),source_ref=ref,
                        pointer=f'/target/baselineExpression/rows/{i}') for i,row in enumerate(rows))
                    signature=json.dumps(rows,sort_keys=True)
                    if (not rows and fetched<total) or (rows and signature in fingerprints):
                        raise ValueError('Baseline expression pagination made no progress')
                    fingerprints.add(signature); fetched+=len(rows); refs.append(ref)
                    if fetched>total:
                        raise ValueError('Baseline expression rows exceed native count')
                collections.append(dict(target_id=target,total_count=total,returned_count=fetched,
                    source_refs=refs,complete_native_collection=fetched==total))
            return dict(expression_collections=collections,
                expression_ranking=rank_expression_material(expression_material,s.ranking.direction) if s.ranking else None,
                interpretation_status='native_baseline_expression_only',
                limitations=['RNA/protein and different datasource/datatype/unit are separate; no cross-assay ranking.',
                             'No high-abundance threshold or safety inference. Specificity is not abundance.'])
        if s.operation in {'retrieve_target_drugs', 'retrieve_tractability'}:
            if not targets:
                selections=[d for d in upstream if d.get('method')=='source_numeric_filter_v1']
                if selections and not any(d['unknown_targets'] for d in selections):
                    return dict(target_material=[],chembl_material=None,selection_empty=True,
                                limitations=['No candidates passed the explicit filter in the retrieved collection.'])
                raise ValueError('Missing verified upstream targets')
            results = []
            for target in (targets if 'open_targets' in s.sources else []):
                query = (ot.KNOWN_DRUGS_QUERY.replace('drugAndClinicalCandidates {', 'id drugAndClinicalCandidates {')
                         if s.operation == 'retrieve_target_drugs' else
                         'query($ensemblId:String!){target(ensemblId:$ensemblId){id tractability{label modality value}}}')
                raw, ref = self._graphql(s, query, {'ensemblId': target})
                record = raw.get('target')
                if not record or record['id'] != target:
                    raise ValueError('Target identity mismatch')
                if s.operation == 'retrieve_target_drugs':
                    collection = record['drugAndClinicalCandidates']
                    results.append({'target_id': target, 'source_ref': ref,
                                    'pointer': '/target/drugAndClinicalCandidates',
                                    'records': collection['rows'], 'total_count': collection['count'],
                                    'truncated': collection['count'] > len(collection['rows'])})
                else:
                    results.append({'target_id': target, 'source_ref': ref,
                                    'pointer': '/target/tractability', 'records': record['tractability']})
            chembl_material = self._chembl_drugs(s, targets) if 'chembl' in s.sources else None
            return {'target_material': results, 'chembl_material': chembl_material,
                    'limitations': ['Source attributes are not disease efficacy or approval.',
                                    'Open Targets drugAndClinicalCandidates has no pagination argument; native response retained.']}
        if s.operation in {'filter_targets', 'rank_targets', 'stratify_target_evidence_layers'}:
            rows = target_inventory(upstream)
            if diseases:
                rows = [r for r in rows if r['disease_id'] in {d['id'] for d in diseases}]
            if s.operation == 'rank_targets':
                rows = rank_inventory(rows, s.ranking)
            if s.operation == 'filter_targets':
                return filter_inventory(rows,s.filters)
            return {'targets': rows, 'method': 'source_evidence_inventory_v1',
                    'limitations': ['Source dimensions are not evidence-strength tiers or clinical efficacy.',
                                    'Ranking applies only to retrieved candidates; unknown values remain last.',
                                    'known_drug_count counts observed distinct Open Targets drug IDs across reported indications; incomplete collections are lower bounds.']}
        # Text search uses explicit entity mentions or their verified canonical names.
        # Do not extract executable queries, IDs, dates or filters from prose descriptions.
        names = [(e.entity_type, resolved.get(e.entity_ref, {}).get('name') or e.mention)
                 for e in entities]
        if not names:
            raise ValueError('Search requires explicit entity_refs')
        if self.human_confirmed_context is not None:
            for e in entities:
                if self.human_confirmed_context.binding_for(e) is not None and e.entity_ref not in resolved:
                    raise ValueError('Human-confirmed scope still requires upstream native identity verification')
        subject_scope=[dict(entity_ref=e.entity_ref,entity_type=e.entity_type,literal_mention=e.mention,
                            search_term=name,canonical_identity_verified=e.entity_ref in resolved)
                       for e,(_,name) in zip(entities,names)]
        packets = []
        phrases = [(kind, '"' + name.replace('"', ' ').replace('\\', ' ') + '"')
                   for kind, name in names]
        for source in s.sources:
            topic = None
            if source == 'clinicaltrials_gov':
                args = {
                    'condition': ' AND '.join(phrase for kind, phrase in phrases if kind == 'disease'),
                    'intervention': ' AND '.join(phrase for kind, phrase in phrases if kind == 'drug'),
                    'term': ' AND '.join(phrase for kind, phrase in phrases
                                         if kind in {'target', 'mechanism', 'tissue'}),
                }
                if s.operation == 'retrieve_trial_terminations':
                    args['overall_statuses'] = ['TERMINATED']
                raw, ref = self._call(s, source, 'search_clinical_trials', args, search_clinical_trials)
            else:
                query = ' AND '.join(phrase for _, phrase in phrases)
                topic = (None if s.operation == 'retrieve_publications' else
                         '(safety OR toxicity OR "adverse event" OR "adverse events")'
                         if s.operation == 'retrieve_safety_evidence' else
                         '(negative OR failure OR ineffective OR terminated)')
                if topic: query += ' AND ' + topic
                raw, ref = self._call(s, source, 'search_publications', {'query': query}, search_publications)
            packets.append({'source_ref': ref, 'source': source, 'material': raw,
                            'subject_scope': subject_scope,
                            'search_scope': {'entity_operator': 'AND', 'topic_expression': topic},
                            'interpretation_status': 'source_material_only'})
        return {'source_material': packets,
                'limitations': ['Keyword hits require evidence organization; do not establish target engagement, safety, negative findings or efficacy.']}

    def _chembl_drugs(self, s, targets):
        mappings, accessions = [], set()
        for target in targets:
            raw, ref = self._graphql(s,
                'query($id:String!){target(ensemblId:$id){id proteinIds{id source}}}', {'id': target})
            record = raw.get('target')
            if not record or record['id'] != target:
                raise ValueError('Protein mapping identity mismatch')
            proteins = [p['id'] for p in record['proteinIds'] if p['source'].lower().startswith('uniprot')]
            accessions.update(proteins)
            mappings.append({'target_id': target, 'protein_accessions': proteins, 'source_ref': ref,
                             'pointer': '/target/proteinIds'})
        if not accessions:
            return {'mappings': mappings, 'status': 'mapping_unavailable',
                    'limitations': ['No native UniProt mapping; not evidence of no drugs.']}
        args = {'path': 'target.json', 'params': {'target_components__accession__in': ','.join(sorted(accessions)),
                                                'limit': 20, 'format': 'json'}}
        raw, ref = self._call(s, 'chembl', 'get_targets_by_accession', args, chembl._get)
        target_rows = raw['targets']
        if not isinstance(target_rows, list) or len(target_rows) > 20:
            raise ValueError('Invalid ChEMBL target mapping page')
        ids = []
        for row in target_rows:
            components = {c['accession'] for c in row['target_components'] if c.get('accession')}
            if not components & accessions:
                raise ValueError('ChEMBL returned a target without the requested protein component')
            ids.append(row['target_chembl_id'])
        result = {'mappings': mappings, 'target_mapping_ref': ref, 'target_records': target_rows,
                  'mapping_truncated': bool(raw['page_meta']['next']) or raw['page_meta']['total_count'] > len(target_rows),
                  'limitations': ['ChEMBL mappings may include complexes or protein families; preserve target_type and components.',
                                  'Mechanism records do not establish disease-specific approval or variant selectivity.']}
        if not ids:
            return dict(result, status='mapping_unavailable')
        args = {'path': 'mechanism.json', 'params': {'target_chembl_id__in': ','.join(sorted(set(ids))),
                                                   'limit': 20, 'format': 'json'}}
        raw, ref = self._call(s, 'chembl', 'get_target_mechanisms', args, chembl._get)
        rows = raw['mechanisms']
        if not isinstance(rows, list) or len(rows) > 20 or any(r['target_chembl_id'] not in ids for r in rows):
            raise ValueError('Invalid ChEMBL mechanism page or target ID')
        return dict(result, status='success', source_ref=ref, records=rows,
                    total_count=raw['page_meta']['total_count'],
                    truncated=bool(raw['page_meta']['next']) or raw['page_meta']['total_count'] > len(rows))

    def run(self, plan: QueryManifestV2, output_path: str | Path | None = None, *, run_id: str | None = None,
            human_confirmed_context: HumanConfirmedEntityContext | None = None,
            research_clarification: HumanResearchClarification | None = None):
        run_id = run_id or str(uuid4())
        if research_clarification is not None:
            research_clarification=research_clarification.for_run(plan.original_question,run_id)
        self.human_confirmed_context = None
        if human_confirmed_context is not None:
            self.human_confirmed_context = human_confirmed_context.for_run(plan.original_question, run_id)
            self.human_confirmed_context.validate_plan(plan)
            if any(self.human_confirmed_context.binding_for(e) is not None and e.entity_id is None for e in plan.entities):
                raise ValueError('Human scope must be explicitly bound in Plan before execution')
        requirement_findings = evaluate_question_requirements(plan, research_clarification, self.human_confirmed_context)
        self.report = {'schema_version': '1.0', 'run_id': run_id, 'started_at': now(),
                       'manifest': plan.model_dump(mode='json'), 'status': 'running',
                       'preflight_issues': _execution_issues(plan, self.human_confirmed_context, requirement_findings),
                       'requirement_findings': [f.model_dump(mode='json') for f in requirement_findings],
                       'steps': [], 'calls': [],
                       'scope': 'Tool execution only; research completion and evidence interpretation not evaluated.'}
        if research_clarification is not None:
            self.report['research_clarification']=research_clarification.model_dump(mode='json')
        if self.human_confirmed_context is not None:
            self.report['human_confirmed_context'] = self.human_confirmed_context.model_dump(mode='json')
        def save():
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding='utf-8')
        if self.report['preflight_issues']:
            self.report.update(status='not_executable', finished_at=now())
            save()
            return self.report
        ancestors, results = {}, {}
        for s in plan.plan_steps:
            ancestors[s.step_id] = set(s.depends_on)
            for dep in s.depends_on:
                ancestors[s.step_id].update(ancestors[dep])
            record = {'step_id': s.step_id, 'operation': s.operation, 'status': 'running',
                      'depends_on': s.depends_on, 'output_names': s.outputs, 'started_at': now()}
            self.report['steps'].append(record)
            if any(results[d]['status'] != 'success' for d in s.depends_on):
                record.update(status='blocked', error='Upstream operation failed')
            else:
                try:
                    refs = set(s.entity_refs)
                    selected = [e for e in plan.entities if e.entity_ref in refs]
                    upstream = [r['result'] for key, r in results.items() if key in ancestors[s.step_id]]
                    record['result'] = self._execute_step(s, selected, upstream)
                    record['status'] = 'success'
                except Exception as exc:
                    record.update(status='error', error=f'{type(exc).__name__}: {exc}')
            record['finished_at'] = now()
            results[s.step_id] = record
            save()
        self.report.update(status='completed' if all(r['status'] == 'success' for r in results.values()) else 'incomplete',
                           finished_at=now())
        save()
        return self.report
