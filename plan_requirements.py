"""Check a small, explicit subset of question requirements against executable fields."""

from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata
from typing import Literal

from execution_capabilities import SOURCES
from planner_acceptance import OPERATION_TABLE
from target_evidence_inventory import POCKET_EVIDENCE_FIELD, PREDICTED_POCKET_FIELD


@dataclass(frozen=True)
class Requirement:
    key: str
    excerpt: str
    status: Literal['covered', 'violated', 'unsupported', 'uncertain']
    detail: str
    correction: str = ''
    operation: str | None = None
    step_id: str | None = None
    origin: str = 'original_question'


_ASSOCIATION = re.compile(
    r'(?:關聯分數|association score)\s*(>=|<=|!=|==|>|<|=|≥|≤|≠|'
    r'嚴格大於|大於等於|大於|小於|至少|at least|greater than|less than)\s*'
    r'([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)'
    r'(?![\d.eE%％])', re.I,
)
_POCKET = re.compile(r'小分子結合口袋|small[- ]molecule binding pocket', re.I)
_EXPRESSION = re.compile(
    r'在哪些正常組織.{0,8}高表現|(?:which|what) normal tissues.{0,45}(?:express|expression)|'
    r'(?:expression|expressed).{0,24}in normal tissues', re.I,
)
_BOOLEAN_UNCLEAR = re.compile(
    r'或|並非|不(?:是|含|包含|具|大於|小於|低於|高於)|未(?:具|有)|沒有|排除|除外|而非|不等於|'
    r'\b(?:or|not|without|except|exclude|unless|between|at most|and/or)\b', re.I,
)
_OTHER_THRESHOLD = re.compile(r'(?:>=|<=|!=|==|[><=≥≤≠])\s*[+-]?[\d.]')
_ISOFORM = re.compile(r'異構型|剪接變體|特定變異|\b(?:isoform|splice variant|specific variant)\b', re.I)
_NAME = r'([A-Za-z][A-Za-z0-9_.\-]*)'
_EXPLICIT_VARIANT = re.compile(
    r'(?:異構型|剪接變體|特定變異|\b(?:isoform|splice variant|specific variant))\s+'
    + r'([A-Za-z][A-Za-z0-9_.\-]*[0-9][A-Za-z0-9_.\-]*)', re.I,
)
_COMPARE = re.compile(r'(?:比較\s*|\bcompare\s+)' + _NAME
                      + r'\s*(?:和|與|\band\b|\bvs\.?\s*)\s*' + _NAME, re.I)
_CONDITIONED_TARGETS = re.compile(r'抑制\s*' + _NAME + r'\s*(?:或|和|與)\s*' + _NAME, re.I)
_CLAUSE_BOUNDARY = re.compile(r'[;；!?。！？]|\.(?=\s|$)')
_UNCLEAR_REQUEST = re.compile(
    r'["“”‘’「」『』]|\b(?:if|unless|whether|never|not|without|except|exclude|avoid|skip|only if)\b|'
    r'不要|不用|不必|無須|無需|毋須|不應|不需|不查|不研究|不考慮|禁止|若|如果|是否'
    r'|\b\w+n[’\x27]t\b', re.I,
)
_FILTER_REQUEST = re.compile(
    r'^\s*(?:(?:列出|篩選|找出).*標的\s*$|'
    r'(?:list|filter|find)\s+[^,;]*?\btargets\s+(?:with|having)\b|which targets\s+(?:have|with)\b)',
    re.I,
)
_EXPRESSION_REQUEST = re.compile(
    r'^\s*(?:它們\s*|[A-Za-z][A-Za-z0-9_.\-]*\s+)?在哪些正常組織.{0,8}高表現\s*$|'
    r'^\s*(?:(?:list|find|retrieve|report|show) (?:the )?(?:[A-Za-z][A-Za-z0-9_.\-]* )?expression in normal tissues'
    r'|(?:which|what) normal tissues.{0,45}(?:express|expression).*)\s*$', re.I,
)


def _request_clause(question, match):
    """Return the containing clause, preserving its request prefix and qualifiers."""
    start = 0
    for boundary in _CLAUSE_BOUNDARY.finditer(question):
        if boundary.start() >= match.end():
            return question[start:boundary.start()]
        if boundary.end() <= match.start():
            start = boundary.end()
    return question[start:]


def _affirmative(question, match, request):
    clause = _request_clause(question, match)
    # Quotes and conditional framing can extend across punctuation. Do not
    # assign obligations when their scope cannot be determined.
    return bool(request.search(clause) and not _UNCLEAR_REQUEST.search(question))


def _ancestors(steps):
    result = {}
    for step in steps:
        result[step.step_id] = set(step.depends_on)
        for parent in step.depends_on:
            result[step.step_id].update(result[parent])
    return result


def _equals(predicate, expected):
    field, operator, value = expected
    names = {field}
    if field == 'overall_association_score':
        names.add('association_score')
    other = predicate.get('value')
    if predicate.get('field') not in names or predicate.get('operator') != operator:
        return False
    if isinstance(value, bool):
        return other is value
    return type(other) in (int, float) and Decimal(str(other)) == value


def _filter_requirements(question, clarification, steps, ancestors, diseases):
    thresholds = list(_ASSOCIATION.finditer(question))
    pocket = _POCKET.search(question)
    if not thresholds and not pocket:
        return []
    trigger = thresholds[0] if thresholds else pocket
    span = trigger.group()
    clause = _request_clause(question, trigger)
    if not _affirmative(question, trigger, _FILTER_REQUEST):
        return [Requirement('filter_logic', span, 'uncertain',
                            'A threshold or pocket mention alone does not establish an inclusion filter.')]
    if len(thresholds) > 1 or _BOOLEAN_UNCLEAR.search(clause) or (
        thresholds and len(_OTHER_THRESHOLD.findall(clause)) > 1
    ) or (pocket and _request_clause(question, pocket) != clause):
        return [Requirement('filter_logic', span, 'uncertain',
                            'Multiple thresholds, disjunction or negation require scope review; an AND filter cannot be assumed equivalent.')]

    providers = [s for s in steps if 'disease_target_association' in OPERATION_TABLE[s.operation].provides]
    provider_scopes = {s.step_id: set(s.entity_refs) & diseases.keys() for s in providers}
    named_refs = {ref for ref, name in diseases.items()
                  if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', clause, re.I)}
    if len(named_refs) != 1:
        return [Requirement('filter_scope', span, 'uncertain',
                            'The inclusion clause cannot be tied to one explicit disease scope.')]
    selected = {pid for pid, refs in provider_scopes.items() if refs == named_refs}
    if any(not refs or (refs & named_refs and refs != named_refs) for refs in provider_scopes.values()):
        return [Requirement('filter_scope', span, 'uncertain',
                            'The association providers do not expose a separable disease branch.')]

    def inputs(step):
        return ancestors[step.step_id] & provider_scopes.keys()

    filters = [s for s in steps if s.operation == 'filter_targets' and inputs(s)
               and inputs(s) <= selected and (set(s.entity_refs) & diseases.keys()) <= named_refs]
    ranks = [s for s in steps if s.operation == 'rank_targets' and inputs(s) & selected
             and (not set(s.entity_refs) & diseases.keys() or set(s.entity_refs) & named_refs)]
    if any(inputs(rank) - selected and (set(rank.entity_refs) & diseases.keys()) != named_refs
           for rank in ranks):
        return [Requirement('filter_scope', span, 'uncertain',
                            'A ranking interleaves disease branches without explicitly selecting a disease.')]
    anchor = next(iter(reversed(filters or ranks or [s for s in providers if s.step_id in selected])), None)

    expected = []
    if thresholds:
        threshold = thresholds[0]
        operator = threshold[1].casefold()
        if operator in {'>', '嚴格大於', '大於', 'greater than'}:
            expected.append(('association_threshold', threshold.group(),
                             ('overall_association_score', 'gt', Decimal(threshold[2])), 'original_question'))
        else:
            return [Requirement('association_threshold', threshold.group(), 'unsupported',
                                'Association filtering currently implements strict gt only.',
                                'Preserve the stated operator and threshold. Use the existing unsupported-operation review path.',
                                step_id=anchor.step_id if anchor else None)]

    pending = []
    if pocket:
        response = clarification.user_response if clarification else ''
        if re.fullmatch(r'納入來源明示的預測口袋[，,]\s*與實驗結構證據分開標示[。.]?', response.strip()):
            expected.append(('pocket', response, (POCKET_EVIDENCE_FIELD, 'is_true', True), 'clarification'))
        elif re.search(r'(?:只納入|僅納入|only)\s*(?:來源明示的)?(?:預測口袋|predicted pockets)', question, re.I):
            expected.append(('pocket', pocket.group(), (PREDICTED_POCKET_FIELD, 'is_true', True), 'original_question'))
        else:
            pending.append(Requirement('pocket', pocket.group(), 'uncertain',
                                       'The source flags distinguish predictions and ligand structures; confirm which qualifies.'))
    if not expected:
        return pending

    def predicates(step):
        values = step.filters.get('all', [])
        return [p for p in values if isinstance(p, dict)] if isinstance(values, list) else []

    # A chained filter inherits its ancestor's selection. Independent filters
    # cannot represent AND, and a downstream rank must consume the selection.
    bundles = []
    for step in filters:
        if any(step.step_id not in ancestors[rank.step_id]
               or (inputs(rank) & selected) - inputs(step)
               for rank in ranks):
            continue
        bundle = [p for f in filters if inputs(f) == inputs(step)
                  and (f.step_id == step.step_id or f.step_id in ancestors[step.step_id])
                  for p in predicates(f)]
        bundles.append((step, bundle))
    for step, bundle in bundles:
        relevant_fields = {'association_score', 'overall_association_score'} if thresholds else set()
        if any(item[0] == 'pocket' for item in expected):
            relevant_fields.update({POCKET_EVIDENCE_FIELD, PREDICTED_POCKET_FIELD})
        if (all(any(_equals(p, item[2]) for p in bundle) for item in expected)
                and all(any(_equals(p, item[2]) for item in expected)
                        for p in bundle if p.get('field') in relevant_fields)):
            return pending + [Requirement(key, excerpt, 'covered', 'Executable filter carries this condition.',
                                         step_id=step.step_id, origin=origin)
                              for key, excerpt, _, origin in expected]
    correction = 'Attach these conditions as executable AND filters on the selected association branch: ' + ', '.join(
        f'{field} {operator} {value}' for _, _, (field, operator, value), _ in expected
    ) + '. Ensure downstream ranking consumes the filtered rows and preserve the required inputs.'
    return pending + [Requirement(
        key, excerpt, 'violated',
        'No matching executable filter chain with the requested predicates and downstream ranking dependency.',
        correction, 'filter_targets', anchor.step_id if anchor else None, origin
    ) for key, excerpt, _, origin in expected]


def assess_plan_requirements(plan, clarification=None, confirmed=None):
    if clarification is not None and clarification.original_question != plan.original_question:
        raise ValueError('Research clarification belongs to a different question')
    if confirmed is not None and confirmed.original_question != plan.original_question:
        raise ValueError('Human confirmation belongs to a different question')
    steps = [step for step in plan.plan_steps
             if step.operation_status == 'approved' and step.operation in SOURCES]
    question = unicodedata.normalize('NFKC', plan.original_question)
    ancestors = _ancestors(plan.plan_steps)
    diseases = {e.entity_ref: unicodedata.normalize('NFKC', e.mention)
                for e in plan.entities if e.entity_type == 'disease'}
    checks = _filter_requirements(question, clarification, steps, ancestors, diseases)
    expression = _EXPRESSION.search(question)
    if expression and not _affirmative(question, expression, _EXPRESSION_REQUEST):
        checks.append(Requirement('normal_tissue_expression', expression.group(), 'uncertain',
                                  'The expression mention is not an unambiguous affirmative request.'))
        expression = None
    if expression:
        candidates = [s for s in steps if s.operation == 'retrieve_normal_tissue_expression']
        relative = clarification and re.fullmatch(
            r'採相對排序[，,]\s*分開呈現不同測量方法[。.]?', clarification.user_response.strip())
        valid = [s for s in candidates if not relative or (
            s.ranking and s.ranking.primary_metric == 'expression_median'
            and s.ranking.direction == 'descending' and not s.ranking.secondary_metrics)]
        subjects = _CONDITIONED_TARGETS.search(question)
        if subjects:
            by_ref = {e.entity_ref: e.mention.casefold() for e in plan.entities if e.entity_type == 'target'}
            needed = {subject.casefold() for subject in subjects.groups()}

            def covers_subjects(subset):
                return needed <= {by_ref.get(ref) for s in subset for ref in s.entity_refs}

            if valid and not covers_subjects(valid):
                valid = []
            if re.search(r'已知的安全風險|\bknown safety risks\b', question, re.I):
                safety = [s for s in steps if s.operation == 'retrieve_safety_evidence']
                checks.append(Requirement(
                    'safety_candidates', subjects.group(), 'covered' if covers_subjects(safety) else 'violated',
                    'Safety candidate retrieval must reference both explicitly named targets; its hits still require review.',
                    'Use the existing safety retrieval binding with the named targets.',
                    'retrieve_safety_evidence', safety[0].step_id if safety else None,
                ))
        checks.append(Requirement(
            'normal_tissue_expression', expression.group(), 'covered' if valid else 'violated',
            'Normal tissue expression requires native retrieval'
            + (' and confirmed within-method relative ranking.' if relative else '.'),
            'Use retrieve_normal_tissue_expression with verified target(s)'
            + (' and descending expression_median without secondary metrics.' if relative else '.'),
            'retrieve_normal_tissue_expression', (valid or candidates)[0].step_id if candidates else None,
        ))
        if relative:
            checks.append(Requirement(
                'expression_order', clarification.user_response, 'covered' if valid else 'violated',
                'Apply native within-source/method/unit relative ranking.',
                'Set descending expression_median on the expression retrieval operation.',
                'retrieve_normal_tissue_expression', (valid or candidates)[0].step_id if candidates else None,
                'clarification',
            ))

    if confirmed:
        consumed = {ref for s in steps for ref in s.entity_refs}
        used_diseases = {ref for s in steps if not s.operation.startswith('resolve_') for ref in s.entity_refs
                         if any(e.entity_ref == ref and e.entity_type == 'disease' for e in plan.entities)}
        confirmed_diseases = {item.entity_id for item in confirmed.entities if item.entity_type == 'disease'}
        for entity in confirmed.entities:
            refs = {e.entity_ref for e in plan.entities if e.mention in entity.mentions
                    and e.entity_type == entity.entity_type and e.entity_id in (None, entity.entity_id)}
            scope_used = bool(refs & consumed)
            # For a single confirmed disease, retaining its unused resolver
            # cannot make a different disease-only retrieval count as coverage.
            if entity.entity_type == 'disease' and len(confirmed_diseases) == 1 and used_diseases:
                scope_used = bool(refs & used_diseases)
            checks.append(Requirement(
                'confirmed_scope:' + entity.entity_id, ' / '.join(entity.mentions),
                'covered' if scope_used else 'violated',
                f'Confirmed scope {entity.entity_id} must be retained and used by the plan.',
                'Retain the confirmed literal entity and its native resolver.',
                origin='human_confirmed_context',
            ))
    comparison = _COMPARE.search(question)
    if comparison and not _affirmative(question, comparison, re.compile(r'^\s*(?:比較|compare\b)', re.I)):
        checks.append(Requirement('comparison_subjects', comparison.group(), 'uncertain',
                                  'The comparison mention is not an unambiguous affirmative request.'))
        comparison = None
    if comparison:
        by_ref = {e.entity_ref: e.mention.casefold() for e in plan.entities}
        used = {by_ref[ref] for s in steps if not s.operation.startswith('resolve_') for ref in s.entity_refs}
        missing = [term for term in comparison.groups() if term.casefold() not in used]
        checks.append(Requirement(
            'comparison_subjects', comparison.group(), 'violated' if missing else 'covered',
            f'Missing explicit comparison subjects in retrieval: {missing}' if missing else 'Both subjects reach an executable consumer.',
            'Preserve both original comparison subjects in their retrieval branches.',
        ))
    molecular = _ISOFORM.search(question)
    molecular_request = re.compile(r'^\s*(?:查找|搜尋|列出|比較|查詢|find\b|search\b|list\b|compare\b|retrieve\b|show\b)', re.I)
    if molecular and not _affirmative(question, molecular, molecular_request):
        checks.append(Requirement('molecular_scope', molecular.group(), 'uncertain',
                                  'The molecular mention is not an unambiguous affirmative request.'))
    elif molecular:
        explicit = _EXPLICIT_VARIANT.search(question)
        if explicit:
            refs = {e.entity_ref for e in plan.entities if e.entity_type == 'target'
                    and e.mention.casefold() == explicit[1].casefold()}
            used = any(refs.intersection(s.entity_refs) for s in steps
                       if not s.operation.startswith('resolve_'))
            if not used:
                checks.append(Requirement(
                    'explicit_variant_subject', explicit.group(), 'violated',
                    'The explicitly labeled variant or isoform is absent from the retrieval scope.',
                    'Keep the original molecular subject in literal search where applicable. Retain unresolved native variant binding.',
                ))
        checks.append(Requirement(
            'molecular_scope', molecular.group(), 'unsupported',
            'No native isoform/variant resolution binding. Literal literature retrieval can preserve the topic; gene-level records cannot establish variant identity or evidence.',
        ))
    exhaustive = re.search(r'全部|所有|完整清單|\b(?:all|every|exhaustive|complete list of)\b', question, re.I)
    bounded = next((s for s in steps if s.operation.startswith('discover_')
                    or set(s.sources) & {'europe_pmc', 'clinicaltrials_gov', 'chembl'}), None)
    if exhaustive and bounded:
        checks.append(Requirement('collection_scope', exhaustive.group(), 'unsupported',
                                  'The available query is a bounded first page or candidate retrieval.',
                                  'Report the source window and retain the exhaustive requirement as unresolved.',
                                  step_id=bounded.step_id))
    verified = re.search(r'已確認|\b(?:verified|confirmed)\b', question, re.I)
    search = next((s for s in steps if set(s.sources) & {'europe_pmc', 'clinicaltrials_gov'}), None)
    if verified and search:
        checks.append(Requirement('verification_scope', verified.group(), 'uncertain',
                                  'Search candidates do not establish the requested verification.',
                                  step_id=search.step_id))
    checks.append(Requirement('unassessed_intent', plan.original_question, 'uncertain',
                              'Only the listed explicit clauses were checked; other intent and biomedical sufficiency require review.'))
    return checks
