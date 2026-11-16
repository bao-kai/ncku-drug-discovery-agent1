"""Native evidence inventory; no clinical scoring or strength classification."""
from copy import deepcopy
import json
import math


NUMERIC_FILTER_FIELDS = {'overall_association_score', 'association_score', 'known_drug_count'}
PREDICTED_POCKET_FIELD = 'predicted_small_molecule_pocket'
POCKET_EVIDENCE_FIELD = 'small_molecule_pocket_evidence'
BOOLEAN_FILTER_FIELDS = {PREDICTED_POCKET_FIELD, POCKET_EVIDENCE_FIELD}


def validate_numeric_filters(filters):
    if not isinstance(filters, dict) or set(filters) != {'all'} or not isinstance(filters['all'], list) or not filters['all']:
        raise ValueError('filter_targets requires filters={"all":[{"field":...,"operator":"gt","value":number}]}')
    for predicate in filters['all']:
        if not isinstance(predicate, dict) or set(predicate) != {'field', 'operator', 'value'}:
            raise ValueError('Numeric filter requires exactly field/operator/value')
        if predicate['field'] in BOOLEAN_FILTER_FIELDS:
            if predicate['operator'] != 'is_true' or predicate['value'] is not True:
                raise ValueError('Pocket evidence filter requires operator=is_true and value=true')
            continue
        if predicate['field'] not in NUMERIC_FILTER_FIELDS or predicate['operator'] != 'gt':
            raise ValueError('Only registered numeric gt and pocket evidence is_true are executable; no pocket predicate for physical_pocket is registered')
        value=predicate['value']
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError('Numeric filter threshold must be finite and must not be bool')


def filter_inventory(rows, filters):
    validate_numeric_filters(filters)
    selected=[]; unknown=[]
    for row in rows:
        states=[]
        for predicate in filters['all']:
            field='overall_association_score' if predicate['field']=='association_score' else predicate['field']
            value=row.get(field)
            if field in BOOLEAN_FILTER_FIELDS:
                if value is not None and not isinstance(value,bool):
                    raise ValueError('Invalid native pocket evidence flag')
                states.append(value)
                continue
            if value is None:
                states.append(None)
            elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'Invalid native numeric value for {field}')
            else:
                states.append(value>predicate['value'])
        if all(s is True for s in states): selected.append(deepcopy(row))
        elif False not in states: unknown.append(deepcopy(row))
    return dict(targets=selected,unknown_targets=unknown,
                selection_input_keys=[[r['disease_id'],r['target_id']] for r in rows],
                method='source_numeric_filter_v1',predicates=deepcopy(filters),
                scope='Retrieved candidate collection only; excluded/unknown rows remain in native source audit.')


def pocket_assessments(packets):
    """Native SM flags only; computational pocket and co-crystal evidence stay separate."""
    labels = ['High-Quality Pocket', 'Med-Quality Pocket', 'Structure with Ligand']
    evidence = {label: [] for label in labels}
    for packet in packets:
        for index, record in enumerate(packet['records']):
            if record.get('modality') != 'SM' or record.get('label') not in evidence:
                continue
            value = record.get('value')
            if value is not None and not isinstance(value, bool):
                raise ValueError('Invalid native tractability boolean')
            evidence[record['label']].append(dict(value=value,source_ref=packet['source_ref'],
                pointer=packet['pointer']+f'/{index}/value'))
    def state(label):
        values=[e['value'] for e in evidence[label]]
        # Conflicting copies are unknown, not silently selected by call order.
        return values[0] if values and all(v is values[0] for v in values) else None
    high, medium = state(labels[0]), state(labels[1])
    predicted = True if high is True or medium is True else False if high is False and medium is False else None
    experimental = state(labels[2])
    inclusive = True if predicted is True or experimental is True else False if predicted is False and experimental is False else None
    return {PREDICTED_POCKET_FIELD: predicted,
            POCKET_EVIDENCE_FIELD: inclusive,
            'predicted_pocket_flags': {label:state(label) for label in labels[:2]},
            'experimental_structure_with_ligand': experimental,
            'pocket_flag_evidence': evidence,
            'pocket_scope': 'SM High-/Med-Quality Pocket are source computational predictions; Structure with Ligand is separate co-crystal evidence. No approval/efficacy inference.'}


def target_inventory(upstream):
    rows = {}
    for data in upstream:
        for row in data.get('targets', []):
            rows[(row['disease_id'], row['target_id'])] = deepcopy(row)
        if data.get('method')=='source_numeric_filter_v1':
            selected={(r['disease_id'],r['target_id']) for r in data['targets']}
            for key in map(tuple,data['selection_input_keys']):
                if key not in selected: rows.pop(key,None)
    for (_, target), row in rows.items():
        drug_packets, tract_packets, chembl_context = {}, {}, []
        for data in upstream:
            for packet in data.get('target_material', []):
                if packet['target_id'] != target:
                    continue
                destination = drug_packets if packet['pointer'].endswith('drugAndClinicalCandidates') else tract_packets
                destination[packet['source_ref']] = deepcopy(packet)
            c = data.get('chembl_material')
            if not c or c.get('status') != 'success':
                continue
            proteins = {p for m in c['mappings'] if m['target_id'] == target for p in m['protein_accessions']}
            mapping_rows = {r['target_chembl_id']: (i, r) for i, r in enumerate(c['target_records'])
                            if proteins & {x.get('accession') for x in r['target_components']}}
            for index, mechanism in enumerate(c['records']):
                if mechanism['target_chembl_id'] not in mapping_rows:
                    continue
                mapping_index, mapping = mapping_rows[mechanism['target_chembl_id']]
                chembl_context.append({'relationship_scope': 'protein_component_mapping',
                    'target_mapping': deepcopy(mapping), 'mapping_source_ref': c['target_mapping_ref'],
                    'mapping_pointer': f'/targets/{mapping_index}', 'mechanism': deepcopy(mechanism),
                    'source_ref': c['source_ref'], 'pointer': f'/mechanisms/{index}',
                    'mapping_truncated': c['mapping_truncated'], 'mechanisms_truncated': c['truncated']})
        drug_ids = set()
        for packet in drug_packets.values():
            for record in packet['records']:
                drug_id = record.get('drug', {}).get('id')
                if not isinstance(drug_id, str) or not drug_id:
                    raise ValueError('Cannot count drug rows without native drug IDs')
                drug_ids.add(drug_id)
        row.update(
            known_drug_count=len(drug_ids) if drug_packets else None,
            known_drug_ids=sorted(drug_ids),
            known_drug_count_scope='Distinct native Open Targets drug IDs in retrieved rows across reported indications; not disease approvals.',
            known_drug_count_is_lower_bound=any(p['truncated'] for p in drug_packets.values()) if drug_packets else None,
            drug_evidence=list(drug_packets.values()), tractability_evidence=list(tract_packets.values()),
            chembl_component_context=chembl_context,
            evidence_dimensions={
                'association': 'source_record_present',
                'open_targets_drugs': ('records_present' if drug_ids else 'no_records_returned') if drug_packets else 'not_collected',
                'tractability': ('records_present' if any(p['records'] for p in tract_packets.values()) else 'no_records_returned') if tract_packets else 'not_collected',
            },
        )
        row.update(pocket_assessments(tract_packets.values()))
    return list(rows.values())


def rank_inventory(rows, ranking):
    """Lexicographic requested metrics; unknowns last, stable native-ID tie break."""
    result = sorted(rows, key=lambda r: (r['disease_id'], r['target_id']))
    for metric in reversed([ranking.primary_metric, *ranking.secondary_metrics]):
        field = 'overall_association_score' if metric == 'association_score' else metric
        present = [r for r in result if r.get(field) is not None]
        missing = [r for r in result if r.get(field) is None]
        present.sort(key=lambda r: r[field], reverse=ranking.direction == 'descending')
        result = present + missing
    return result


def expression_comparison_scope(row):
    """Tissue is the varying axis; cell types and native source contexts stay separate."""
    tissue = row.get('tissueBiosample') or row.get('tissueBiosampleFromSource')
    cell = row.get('celltypeBiosample') or row.get('celltypeBiosampleFromSource')
    return (row.get('targetId'), row.get('datasourceId'), row.get('datatypeId'), row.get('unit'),
            row.get('targetFromSourceId'), 'tissue' if tissue else 'celltype_only',
            json.dumps(cell,sort_keys=True), row.get('celltypeBiosampleFromSource'),
            row.get('qualityControls'))


def rank_expression_material(material, direction):
    """Sort source medians without pooling measurements or inventing a high cutoff."""
    groups={}
    for item in material:
        raw=item['record']; scope=expression_comparison_scope(raw)
        key=tuple(json.dumps(v,sort_keys=True) for v in scope)
        comparable=all(raw.get(k) for k in ['targetId','datasourceId','datatypeId','unit'])
        if not comparable:
            key=(*key,item['source_ref'],item['pointer'])
        scope_fields=['target_id','datasource_id','datatype_id','unit','target_from_source_id',
                      'annotation_scope','celltype_biosample','celltype_from_source','quality_controls']
        groups.setdefault(key,dict(scope=dict(zip(scope_fields,scope)),comparable=comparable,rows=[]))['rows'].append(deepcopy(item))
    result=[]
    for key in sorted(groups):
        group=groups[key]
        def value(item):
            v=item['record'].get('median')
            return v if group['comparable'] and not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) else None
        ordered=sorted(group['rows'],key=lambda x:(json.dumps(x['record'].get('tissueBiosample'),sort_keys=True),
            str(x['record'].get('tissueBiosampleFromSource')),x['source_ref'],x['pointer']))
        known=[r for r in ordered if value(r) is not None]; unknown=[r for r in ordered if value(r) is None]
        known.sort(key=value,reverse=direction=='descending')
        group['rows']=[dict(source_ref=r['source_ref'],pointer=r['pointer'],
                           tissue_biosample=deepcopy(r['record'].get('tissueBiosample')),
                           tissue_from_source=r['record'].get('tissueBiosampleFromSource'),
                           source_median=value(r),position=i+1 if value(r) is not None else None)
                       for i,r in enumerate(known+unknown)]
        result.append(group)
    return dict(metric='expression_median',native_field='median',direction=direction,groups=result,
        interpretation='Relative native expression within comparable source/method/unit/cell-type scope; no absolute high-expression threshold or clinical harm inference.',
        tie_behavior='Equal medians retain stable native tissue/source-pointer order; position is display order only.',
        missing_behavior='Missing/nonfinite/non-numeric median or incomplete comparison metadata stays unknown, after known values; no imputation or averaging.')
