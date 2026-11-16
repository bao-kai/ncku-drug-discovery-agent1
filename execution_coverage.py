"""Native execution coverage and exact cached-material references, not science judgments."""
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5
from types import SimpleNamespace
import re


def _field_value_state(value):
    if value is None:
        return 'null'
    if value is False:
        return 'false'
    if not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0:
        return 'zero'
    if isinstance(value, str) and not value:
        return 'empty_string'
    if isinstance(value, list) and not value:
        return 'empty_array'
    if isinstance(value, dict) and not value:
        return 'empty_object'
    return 'value'


def _field_options(coverage_id, rows, run_id):
    # Reuse existing native evidence units; never infer paths from gap prose.
    from evidence_selection import build_catalog
    group = SimpleNamespace(snapshot_ids=[r.snapshot_id for r in rows], context_snapshot_ids=[])
    units = build_catalog(group, {r.snapshot_id:r for r in rows}, [], run_id)
    paths = sorted({u['json_pointer'] for u in units.values() if u['reference_scope']=='raw_record'})
    return [dict(field_id=str(uuid5(NAMESPACE_URL, coverage_id + '/field/' + p)),
                 json_pointer=p, contract='native_record_inventory_for_this_material') for p in paths]


def coverage_index(execution, records, groups):
    by_call = defaultdict(list)
    owners = defaultdict(list)
    for record in records:
        by_call[record.execution_origin['call_id']].append(record)
    for group in groups:
        for sid in group.snapshot_ids:
            owners[sid].append(group.group_id)
    index = []

    def add(call, subject, material, pointer, rows, total=None, partial=False, status=None):
        ids = [r.snapshot_id for r in rows]
        state = status or ('partial' if partial else 'empty_result' if total == 0 else 'collected')
        index.append(dict(
            coverage_id=str(uuid5(NAMESPACE_URL, f'{execution["run_id"]}/{call["call_id"]}/{subject}/{material}/{pointer}')),
            subject_ref=subject, source=call['source'], material_kind=material,
            state=state, call_id=call['call_id'], json_pointer=pointer,
            source_url=call.get('source_url'), snapshot_ids=ids,
            group_ids=sorted({g for sid in ids for g in owners[sid]}),
            returned_count=len(rows), total_count=total,
            record_pointers=[r.execution_origin['json_pointer'] for r in rows],
            scope='Native returned collection only; field completeness and scientific sufficiency are unverified.'))
        index[-1]['field_options'] = _field_options(index[-1]['coverage_id'], rows, execution['run_id'])

    steps = {s['step_id']: s for s in execution['manifest'].get('plan_steps', [])}
    for call in execution['calls']:
        raw = call.get('raw_response') or {}
        rows = by_call[call['call_id']]
        variables = call.get('arguments', {}).get('variables', {})
        disease, target = raw.get('disease') or {}, raw.get('target') or {}
        if call['status'] != 'success':
            subject = ('target:' + variables['ensemblId'] if variables.get('ensemblId') else
                       'disease:' + variables['efoId'] if variables.get('efoId') else 'call:' + call['call_id'])
            add(call, subject, steps.get(call['step_id'], {}).get('operation', call['tool']), '', [],
                status='empty_result' if call['status'] == 'not_found' else 'source_error')
        elif 'associatedTargets' in disease:
            data = disease['associatedTargets']
            add(call, 'disease:' + disease['id'], 'association_collection', '/disease/associatedTargets',
                rows, data['count'], data['count'] > len(data['rows']))
            for row in rows:
                tid = row.raw_record['target']['id']
                add(call, 'pair:' + disease['id'] + ':' + tid, 'association',
                    row.execution_origin['json_pointer'], [row], 1)
        elif 'drugAndClinicalCandidates' in target:
            data = target['drugAndClinicalCandidates']
            add(call, 'target:' + target['id'], 'target_drugs', '/target/drugAndClinicalCandidates',
                rows, data['count'], data['count'] > len(data['rows']))
        elif 'tractability' in target:
            add(call, 'target:' + target['id'], 'tractability', '/target/tractability',
                rows, len(target['tractability']))
        elif 'baselineExpression' in target:
            data=target['baselineExpression']
            add(call,'target:'+target['id'],'normal_tissue_expression','/target/baselineExpression',
                rows,data['count'],len(data['rows'])<data['count'])
        elif call['source'] == 'chembl' and 'mechanisms' in raw:
            total = raw['page_meta']['total_count']
            add(call, 'call:' + call['call_id'], 'chembl_mechanisms', '/mechanisms', rows, total,
                total > len(raw['mechanisms']))
        elif rows or 'records' in raw:
            total = raw.get('total_count', len(rows))
            add(call, 'call:' + call['call_id'], call['tool'],
                '/raw_record' if raw.get('raw_record') else '/records', rows, total,
                bool(raw.get('truncated')) or total > len(rows))
    called = {c['step_id'] for c in execution['calls']}
    for step in execution['steps']:
        spec = steps.get(step['step_id'], {})
        if step['step_id'] in called or step['status'] == 'success':
            continue
        for source in spec.get('sources', []):
            call = dict(call_id='unexecuted:' + step['step_id'], source=source)
            add(call, 'step:' + step['step_id'], spec['operation'], '', [], status='not_collected')
    return index


def validate_coverage_selection(selection, index):
    by_id = {c['coverage_id']: c for c in index}
    for gap in selection.review_flags:
        for ref in gap.coverage_refs or []:
            entry = by_id.get(ref.coverage_id)
            if entry is None:
                raise ValueError('Unknown coverage_id')
            if any(getattr(ref, k) != entry[k] for k in ('subject_ref', 'source', 'material_kind')):
                raise ValueError('Coverage subject/source/material scope mismatch')
            if ref.required_pointer is not None and not ref.required_pointer.startswith('/'):
                raise ValueError('Coverage required_pointer must be relative to a native record and start with /')
            if ref.field_id is not None:
                field = next((f for f in entry.get('field_options', []) if f['field_id']==ref.field_id), None)
                if field is None:
                    raise ValueError('Unknown field_id for this coverage material')
                if ref.required_pointer is not None and ref.required_pointer != field['json_pointer']:
                    raise ValueError('field_id conflicts with required_pointer; no pointer rewriting is allowed')


def reconcile_gap(gap, index, group, records):
    """Preserve prose; classify only explicitly selected native scope and fields."""
    from evidence_preparation import resolve_json_pointer
    by_id = {c['coverage_id']: c for c in index}
    record_map = {r.snapshot_id: r for r in records}
    local = set(group.snapshot_ids + group.context_snapshot_ids)
    decisions = []
    for ref in gap.get('coverage_refs') or []:
        entry = by_id.get(ref['coverage_id'])
        if entry is None or any(ref[k] != entry[k] for k in ('subject_ref', 'source', 'material_kind')):
            raise ValueError('Invalid saved coverage reference')
        ids = entry['snapshot_ids']
        present, absent, unverified, observations = [], [], [], []
        pointer = ref.get('required_pointer')
        field_id = ref.get('field_id')
        if field_id:
            field = next((f for f in entry.get('field_options', []) if f['field_id']==field_id), None)
            if field is None or pointer is not None and pointer != field['json_pointer']:
                raise ValueError('Invalid saved field_id or conflicting pointer')
            pointer = field['json_pointer']
        known = pointer is not None and any(f['json_pointer']==pointer for f in entry.get('field_options', []))
        syntactically_valid = pointer is None or pointer.startswith('/') and not re.search(r'~(?![01])',pointer)
        field_reference_state = ('not_requested' if pointer is None else
                                 'inventory_field' if known else
                                 'unverified_field_reference' if syntactically_valid else 'invalid_field_reference')
        for sid in ids:
            if pointer is None:
                present.append(sid)
                continue
            if syntactically_valid:
                try:
                    value = resolve_json_pointer(record_map[sid].raw_record, pointer)
                    observed = _field_value_state(value)
                except ValueError:
                    observed = 'absent' if known else 'unverified_position'
            else:
                observed = 'invalid_position'
            observations.append(dict(snapshot_id=sid, state=observed))
            if observed in {'value','false','zero'}:
                present.append(sid)
            if observed == 'absent':
                absent.append(sid)
            if observed in {'unverified_position','invalid_position'}:
                unverified.append(sid)
        if pointer is not None and not known and observations and not unverified:
            field_reference_state = 'observed_native_pointer'
        observed_states = {o['state'] for o in observations}
        detail_state = ('not_specified' if pointer is None else
                        'no_returned_records' if not ids else
                        field_reference_state if unverified and not present else
                        'all_returned_records_present' if len(present)==len(ids) else
                        'source_' + next(iter(observed_states)) if len(observed_states)==1 else
                        'mixed_field_states_requires_review')
        state = entry['state']
        if state == 'collected':
            state = (field_reference_state if unverified else
                     'known_field_absent' if absent and len(absent)==len(ids) else
                     'source_field_state_requires_review' if pointer is not None and len(present)!=len(ids) else
                     'collected_in_group' if set(present).issubset(local) else 'collected_elsewhere')
        decisions.append(dict(
            coverage_id=entry['coverage_id'], subject_ref=entry['subject_ref'], source=entry['source'],
            material_kind=entry['material_kind'], state=state, collection_state=entry['state'],
            call_id=entry['call_id'], json_pointer=entry['json_pointer'],
            requested_field_id=field_id, required_pointer=ref.get('required_pointer'), resolved_pointer=pointer,
            field_reference_state=field_reference_state, detail_state=detail_state,
            field_observations=observations, matched_snapshot_ids=present,
            absent_field_snapshot_ids=absent, unverified_field_snapshot_ids=unverified,
            reuse_snapshot_ids=[sid for sid in present if sid not in local],
            judgment='Collection/field availability only; requested conditions and scientific sufficiency require review.'))
    return dict(original_flag=gap, state='typed_coverage' if decisions else 'coverage_unmapped_requires_review',
                references=decisions)
