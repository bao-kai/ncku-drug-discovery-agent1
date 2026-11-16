"""Reproduce executed filters and expose bounded outcomes, including zero matches."""
from copy import deepcopy
import json
from target_evidence_inventory import target_inventory,filter_inventory


def assessed_filters(execution):
    specs={s['step_id']:s for s in execution['manifest'].get('plan_steps',[])}
    steps={s['step_id']:s for s in execution['steps']};result=[]
    def ancestors(sid):
        found=set();todo=list(specs[sid].get('depends_on',[]))
        while todo:
            dep=todo.pop()
            if dep not in found:
                found.add(dep);todo.extend(specs.get(dep,{}).get('depends_on',[]))
        return found
    for sid,spec in specs.items():
        if spec.get('operation')!='filter_targets':continue
        step=steps.get(sid,{});parents=ancestors(sid)
        calls=[c for c in execution['calls'] if c['step_id'] in parents]
        assessment=dict(filter_step_id=sid,source_run_id=execution['run_id'],state='not_evaluated',
            predicate=deepcopy(spec.get('filters',{})),scope='Retrieved candidate collection only; no whole-database absence claim.',
            source_calls=[{k:c[k] for k in ['call_id','source','source_url','status'] if k in c} for c in calls])
        if execution['status']=='completed' and step.get('status')=='success' and calls and all(c['status']=='success' for c in calls) and all(steps.get(d,{}).get('status')=='success' for d in parents):
            upstream=[s['result'] for s in execution['steps'] if s['step_id'] in parents]
            rows=target_inventory(upstream)
            resolved=[e for d in upstream for e in d.get('entities',[]) if e['entity_ref'] in spec.get('entity_refs',[]) and e['entity_type']=='disease']
            if resolved:rows=[r for r in rows if r['disease_id'] in {e['id'] for e in resolved}]
            try:
                reproduced=filter_inventory(rows,spec['filters'])
                if json.dumps(reproduced,sort_keys=True)!=json.dumps(step['result'],sort_keys=True):raise ValueError('Saved filter differs from upstream source evaluation')
                qualified=len(reproduced['targets']);unknown=len(reproduced['unknown_targets'])
                assessment.update(state='unresolved_inputs' if unknown else 'evaluated_matches' if qualified else 'evaluated_empty',
                    evaluated_count=len(rows),qualified_count=qualified,unknown_count=unknown,
                    excluded_count=len(rows)-qualified-unknown,result_ref=f'/steps/{execution["steps"].index(step)}/result',
                    qualified_targets=deepcopy(reproduced['targets']),unknown_targets=deepcopy(reproduced['unknown_targets']),
                    input_keys=deepcopy(reproduced['selection_input_keys']),reproduced_from_sources=True)
            except (ValueError,KeyError,TypeError) as exc:assessment['evaluation_error']=str(exc)
        result.append(assessment)
    return result


def group_assessed_result(preparation,group):
    from evidence_selection import group_selection_scope
    scope=group_selection_scope(preparation,group)
    if scope is None:return None
    wanted={r['filter_step_id'] for r in scope['candidate_memberships']}
    entries=[a for a in preparation.query_context.get('assessed_filters',[]) if a['filter_step_id'] in wanted]
    proven=len(entries)==len(wanted) and all(a['state'] in ['evaluated_empty','evaluated_matches'] and a.get('reproduced_from_sources') for a in entries)
    state='evaluated_empty' if proven and all(a['state']=='evaluated_empty' for a in entries) else 'background_only' if proven and scope['membership']=='excluded' else 'evaluation_requires_material'
    return dict(state=state,allow_empty_response=state in ['evaluated_empty','background_only'],assessments=deepcopy(entries))


def already_assessed_filter_input(assessed,decision):
    """Keep bounded scope flags without treating them as new material requests."""
    if (not assessed or assessed['state'] != 'evaluated_empty' or not decision['references']
            or decision['original_flag'].get('followup_request_id')):
        return False
    fields={p['field'] for a in assessed['assessments'] for p in a['predicate'].get('all',[])}
    pointers=set()
    if fields.intersection({'overall_association_score','association_score'}):pointers.add('/score')
    if fields.intersection({'small_molecule_pocket_evidence','predicted_small_molecule_pocket'}):pointers.add('/tractability')
    calls={c['call_id'] for a in assessed['assessments'] for c in a['source_calls']}
    return all(r['call_id'] in calls and (r.get('resolved_pointer') in pointers or
               r.get('collection_state')=='partial') and
               r.get('detail_state')=='all_returned_records_present' for r in decision['references'])


def filter_answer(execution,organization,execution_file):
    """Render evaluated result facts; never turn bounded absence into database-wide absence."""
    assessments=organization.get('assessed_filters',[])
    status='資料流程與引用檢查完成' if organization['status']=='references_validated' else '流程未完成'
    lines=[execution['manifest']['original_question'],'',f'流程狀態：{status}。']
    for a in assessments:
        names={'overall_association_score':'關聯分數','association_score':'關聯分數',
            'known_drug_count':'已取回的原生藥物數','small_molecule_pocket_evidence':'來源小分子口袋證據（預測或共結晶）',
            'predicted_small_molecule_pocket':'來源預測小分子口袋'}
        conditions=[names.get(p['field'],p['field'])+('嚴格大於 '+str(p['value']) if p['operator']=='gt' else '為 true') for p in a['predicate'].get('all',[])]
        lines+=['','篩選條件：'+'，且'.join(conditions)+'。']
        if not a.get('reproduced_from_sources'):
            lines.append('尚未完成可由來源重現的篩選判定；不宣稱零筆符合。');continue
        lines.append(f'本次取回並評估 {a["evaluated_count"]} 筆候選，符合 {a["qualified_count"]} 筆，因必要值未知而未能判定 {a["unknown_count"]} 筆，排除 {a["excluded_count"]} 筆。')
        for row in a['qualified_targets']:
            lines.append(f'- {row.get("target_symbol") or row["target_id"]} ({row["target_id"]})：association={row.get("overall_association_score")}；預測旗標={json.dumps(row.get("predicted_pocket_flags"),ensure_ascii=False)}；實驗共結晶旗標={row.get("experimental_structure_with_ligand")}。')
        lines.append('此判定只適用於已取回集合；不表示全資料庫不存在其他合格標的。')
        if any(p.get('field')=='small_molecule_pocket_evidence' for p in a['predicate'].get('all',[])):
            lines.append('口袋資格納入來源 SM High-/Med-Quality Pocket 預測或 Structure with Ligand 共結晶旗標；兩類證據分開保留，不把預測稱為實驗證實。')
    for step in execution['steps']:
        for collection in step.get('result',{}).get('collections',[]):
            lines.append(f'來源集合：取回 {collection["returned_count"]}/{collection["total_count"]} 筆，truncated={collection["truncated"]}。')
    flags=[f for g in organization.get('groups',[]) for f in (g['result'].get('structured_output') or {}).get('review_flags',[])]
    if flags:lines.append(f'整理仍保留 {len(flags)} 項待核對材料標記，詳見整理報告；不視為已解決。')
    lines+=['',f'[本次原始來源與逐值指標]({execution_file})']
    urls=list(dict.fromkeys(c.get('source_url') for c in execution['calls'] if c.get('source_url')))
    lines.extend(f'- [來源 API {i+1}]({url})' for i,url in enumerate(urls))
    return '\n'.join(lines)+'\n'
