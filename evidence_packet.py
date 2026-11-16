"""Self-contained, reversible tables for organizer input. No evidence pruning or retrieval."""
from copy import deepcopy
import json


CONTRACT = 'references_v1'
INSTRUCTIONS = (
    '\nInput contract references_v1 is a lossless, SELF-CONTAINED representation, not external files. All following payload fields are inside data. '
    'records.rows are [snapshot_id, record_id, metadata_index, native_pointer, layout_index, layout_row_index]. '
    'Merge records.metadata[metadata_index], restoring execution_origin.json_pointer from native_pointer. '
    'native_layouts have fields and columns: a column contains constant, values[row], or dictionary[indices[row]]. '
    'All native values and nulls are present in this packet; different layouts preserve absent fields. '
    'catalog groups share header and value_source. Each entry is [zero_based_evidence_ordinal, record_indices, optional_literal_value]. '
    'Evidence ordinal n names E(n+1), unless evidence_keys is supplied. record_indices address records.rows. '
    'For value_source=record, obtain the unit value via header.json_pointer in the FIRST referenced native record; '
    'literal means entry[2], none means no value (absent). record_count is the membership length. '
    'scalar_value_counts_refs[field] holds [record_index,count], reading that native record field for the value. '
    'relative_expression_rankings groups use row_refs=[record_index,position]; tissue/median are native fields, '
    'and null position means interpreted median remains unknown. Python already computed this order; do not rerank. '
    'query_context.execution_coverage aliases the single top-level execution_coverage list. Global availability stays visible. '
    'source_subjects record_indices use this same record order and are a SOURCE-derived namespace, distinct from Planner entities. '
    'selection_scope states result eligibility; excluded/unknown candidates remain source background, never qualified results.\n'
)


def _json(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def _pointer(value,path):
    for key in path.lstrip('/').split('/') if path else []:
        key=key.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def _index(items,index):
    if isinstance(index,bool) or not isinstance(index,int) or index<0 or index>=len(items):
        raise ValueError('Packet reference is outside its table')
    return items[index]


def _columns(rows,fields):
    result=[]
    for field in fields:
        values=[r[field] for r in rows];dictionary=[];lookup={};indices=[]
        for value in values:
            marker=_json(value)
            if marker not in lookup:lookup[marker]=len(dictionary);dictionary.append(value)
            indices.append(lookup[marker])
        options=[dict(values=values),dict(dictionary=dictionary,indices=indices)]
        if len(dictionary)==1:options.append(dict(constant=dictionary[0]))
        result.append(min(options,key=lambda v:len(_json(v).encode('utf-8'))))
    return result


def pack_input(data):
    """Encode only redundant views; every lookup can be resolved within this object."""
    out=deepcopy(data);original=data['records'];layouts=[];layout_rows=[];layout_map={}
    metadata=[];meta_map={};rows=[]
    for record in original:
        raw=record['raw_record'];fields=tuple(raw)
        if fields not in layout_map:
            layout_map[fields]=len(layouts);layouts.append(dict(fields=list(fields)));layout_rows.append([])
        layout=layout_map[fields];native_row=len(layout_rows[layout]);layout_rows[layout].append(raw)
        common={k:deepcopy(v) for k,v in record.items() if k not in ['snapshot_id','record_id','raw_record']}
        origin=common.get('execution_origin');pointer=origin.pop('json_pointer',None) if isinstance(origin,dict) else None
        marker=_json(common)
        if marker not in meta_map:meta_map[marker]=len(metadata);metadata.append(common)
        rows.append([record['snapshot_id'],record['record_id'],meta_map[marker],pointer,layout,native_row])
    for layout,natives in zip(layouts,layout_rows):
        layout.update(row_count=len(natives),columns=_columns(natives,layout['fields']))
    out['records']=dict(metadata=metadata,native_layouts=layouts,rows=rows)
    groups=[];headers={};keys=list(data['catalog'])
    for ordinal,(key,unit) in enumerate(data['catalog'].items()):
        if unit['record_count']!=len(unit['record_indices']):raise ValueError('Catalog record count does not match memberships')
        header={k:v for k,v in unit.items() if k not in ['value','record_indices','record_count']}
        source='record' if 'value' in unit and unit['reference_scope']=='raw_record' else 'literal' if 'value' in unit else 'none'
        marker=_json([header,source])
        if marker not in headers:
            headers[marker]=len(groups);groups.append(dict(header=header,value_source=source,entries=[]))
        entry=[ordinal,unit['record_indices']]
        if source=='literal':entry.append(unit['value'])
        groups[headers[marker]]['entries'].append(entry)
    out['catalog']=dict(groups=groups)
    if keys!=[f'E{i+1}' for i in range(len(keys))]:out['catalog']['evidence_keys']=keys
    counts=data['group'].get('scalar_value_counts',{})
    lookups={field:{_json(r['raw_record'][field]):i for i,r in reversed(list(enumerate(original))) if field in r['raw_record']} for field in counts}
    out['group'].pop('scalar_value_counts',None)
    out['group']['scalar_value_counts_refs']={field:[[lookups[field][_json(row['value'])],row['count']] for row in entries] for field,entries in counts.items()}
    context=out['query_context']
    if 'execution_coverage' in context:
        if context['execution_coverage']!=data.get('execution_coverage'):raise ValueError('Coverage copies disagree')
        context['execution_coverage']={'$packet_ref':'/execution_coverage'}
    by_origin={(r.get('execution_origin') or {}).get('call_id','')+'\0'+(r.get('execution_origin') or {}).get('json_pointer',''):i for i,r in enumerate(original)}
    for ranking in context.get('relative_expression_rankings',[]):
        for group in ranking['result']['groups']:
            refs=[]
            for row in group.pop('rows'):
                index=by_origin[row['source_ref']+'\0'+row['pointer']];raw=original[index]['raw_record']
                expected=dict(source_ref=row['source_ref'],pointer=row['pointer'],position=row['position'],
                    tissue_biosample=raw.get('tissueBiosample'),tissue_from_source=raw.get('tissueBiosampleFromSource'),
                    source_median=raw.get('median') if row['position'] is not None else None)
                if _json(expected)!=_json({k:row[k] for k in expected}) or set(row)!=set(expected):
                    raise ValueError('Ranking row cannot be losslessly referenced to its native record')
                refs.append([index,row['position']])
            group['row_refs']=refs
    return dict(input_contract=CONTRACT,data=out)


def unpack_input(packet):
    """Rebuild the expanded wire view, including legacy units, values and statistics."""
    if packet.get('input_contract')!=CONTRACT:return deepcopy(packet)
    out=deepcopy(packet['data']);table=out['records'];records=[]
    for snapshot,record_id,meta,pointer,layout_id,row_id in table['rows']:
        common=deepcopy(_index(table['metadata'],meta));layout=_index(table['native_layouts'],layout_id)
        if row_id<0 or row_id>=layout['row_count']:raise ValueError('Native row reference outside layout')
        native={}
        for field,column in zip(layout['fields'],layout['columns']):
            native[field]=(column['constant'] if 'constant' in column else _index(column['values'],row_id) if 'values' in column
                           else _index(column['dictionary'],_index(column['indices'],row_id)))
        if pointer is not None:common['execution_origin']['json_pointer']=pointer
        records.append(dict(**common,snapshot_id=snapshot,record_id=record_id,raw_record=native))
    out['records']=records;catalog={};key_table=out['catalog'].get('evidence_keys')
    for group in out['catalog']['groups']:
        for entry in group['entries']:
            ordinal,indices=entry[:2];unit=dict(**group['header'],record_indices=indices,record_count=len(indices))
            for i in indices:_index(records,i)
            if group['value_source']=='literal':unit['value']=entry[2]
            elif group['value_source']=='record':unit['value']=_pointer(_index(records,indices[0])['raw_record'],unit['json_pointer'])
            catalog[ordinal]=(key_table[ordinal] if key_table else f'E{ordinal+1}',unit)
    out['catalog']=dict(catalog[i] for i in sorted(catalog))
    refs=out['group'].pop('scalar_value_counts_refs')
    out['group']['scalar_value_counts']={field:[dict(value=_index(records,i)['raw_record'][field],count=count) for i,count in entries] for field,entries in refs.items()}
    context=out['query_context']
    if context.get('execution_coverage')=={'$packet_ref':'/execution_coverage'}:
        context['execution_coverage']=deepcopy(out['execution_coverage'])
    for ranking in context.get('relative_expression_rankings',[]):
        for group in ranking['result']['groups']:
            group['rows']=[]
            for index,position in group.pop('row_refs'):
                record=_index(records,index);raw=record['raw_record'];origin=record['execution_origin']
                group['rows'].append(dict(source_ref=origin['call_id'],pointer=origin['json_pointer'],
                    tissue_biosample=deepcopy(raw.get('tissueBiosample')),tissue_from_source=raw.get('tissueBiosampleFromSource'),
                    source_median=raw.get('median') if position is not None else None,position=position))
    return out
