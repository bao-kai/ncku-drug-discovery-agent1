"""Hosted natural-language Plan -> tools -> organization -> bounded follow-up."""
from pathlib import Path
from time import perf_counter
import json
from uuid import uuid4
from human_entity_context import HumanConfirmedEntityContext, HumanResearchClarification

from langchain_core.messages import AIMessage
from natural_language_planner import DirectJsonPlanAgent, DirectJsonPatchAgent, NaturalLanguageResearchPlanner
from plan_execution import PlanExecutor
from execution_organization import organize_execution
from nim_inference import stream_nim_completion


class HostedPlannerModel:
    def __init__(self, client, output_dir):
        self.client, self.output_dir, self.calls = client, Path(output_dir), 0

    def invoke(self, messages, config=None):
        self.calls += 1
        prefix = self.output_dir / f'planner_call_{self.calls}'
        prefix.with_suffix('.input.json').write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')
        result = stream_nim_completion(self.client, messages)
        prefix.with_suffix('.result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        if result['status'] != 'response_received':
            raise RuntimeError(f'Planner generation failed: {result["status"]}')
        return AIMessage(content=result['response_text'])


def run_query(question, client, output_dir, *, acceptance_case=None,
              human_confirmed_context: HumanConfirmedEntityContext | None = None,
              research_clarification: HumanResearchClarification | None = None):
    run_id = str(uuid4())
    if human_confirmed_context is not None:
        human_confirmed_context = human_confirmed_context.for_run(question, run_id)
    if research_clarification is not None:
        research_clarification=research_clarification.for_run(question,run_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError('Use an empty output directory to preserve previous run evidence')
    start = perf_counter()
    summary = {'question': question, 'status': 'running', 'stages': {}, 'output_dir': str(output_dir.resolve())}
    summary['run_id'] = run_id
    if research_clarification is not None:
        (output_dir/'research_clarification.json').write_text(research_clarification.model_dump_json(indent=2),encoding='utf-8')
        summary['research_clarification']=research_clarification.model_dump(mode='json')
    if human_confirmed_context is not None:
        (output_dir/'human_confirmed_context.json').write_text(
            human_confirmed_context.model_dump_json(indent=2), encoding='utf-8')
        summary['scope_basis'] = human_confirmed_context.approval_basis
    def save():
        summary['elapsed_seconds'] = round(perf_counter() - start, 3)
        (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    model = HostedPlannerModel(client, output_dir)
    planner = NaturalLanguageResearchPlanner(agent=DirectJsonPlanAgent(model), patch_agent=DirectJsonPatchAgent(model),
                                            execution_mode=True, capture_model_calls=False)
    report = planner.run_with_report(question, acceptance_case=acceptance_case,
                                     checkpoint_path=output_dir/'planner.report.json', max_attempts=4,
                                     human_confirmed_context=human_confirmed_context,
                                     research_clarification=research_clarification)
    summary['stages']['planner'] = {'status': report.status, 'execution_checked': report.execution_checked,
                                  'calls': model.calls, 'elapsed_seconds': report.total_duration_seconds}
    evaluated = next((attempt.operation_rule_evaluation for attempt in reversed(report.attempts)
                      if attempt.operation_rule_evaluation is not None), None)
    if evaluated is not None:
        summary['requirement_findings'] = [
            item.model_dump(mode='json')
            for item in evaluated.findings
            if item.requirement_status is not None
        ]
    if report.final_plan is None:
        summary['status'] = 'blocked_at_planner'
        save()
        return summary
    (output_dir/'plan.json').write_text(report.final_plan.model_dump_json(indent=2), encoding='utf-8')
    before = perf_counter()
    execution_path = output_dir/'execution.json'
    execution = PlanExecutor().run(report.final_plan, execution_path, run_id=run_id,
                                   human_confirmed_context=human_confirmed_context,
                                   research_clarification=research_clarification)
    summary['stages']['execution'] = {'status': execution['status'], 'calls': len(execution['calls']),
                                    'elapsed_seconds': round(perf_counter() - before, 3)}
    save()
    if execution['status'] != 'completed':
        summary['status'] = 'blocked_at_execution'
        save()
        return summary
    before = perf_counter()
    result = organize_execution(execution, client, output_dir/'organization.json', execution_path=execution_path)
    summary['stages']['organization'] = {'status': result['status'], 'groups': len(result['groups']),
                                       'elapsed_seconds': round(perf_counter() - before, 3),
                                       'followup': result.get('followup', {})}
    summary['status'] = result['status']
    summary['delivery'] = {'execution_file': str(execution_path.resolve()),
                           'organization_file': str((output_dir/'organization.json').resolve()),
                           'analysis_step_ids': [s['step_id'] for s in execution['steps']
                                                 if s['operation'] in {'stratify_target_evidence_layers','rank_targets'}],
                           'scope': 'Source inventory and references, not clinical efficacy or expert scientific approval'}
    if result.get('assessed_filters'):
        from filter_assessment import filter_answer
        answer_path=output_dir/'answer.md'
        answer_path.write_text(filter_answer(execution,result,str(execution_path.resolve())),encoding='utf-8')
        summary['delivery']['answer_file']=str(answer_path.resolve())
    save()
    return summary
