"""Plan-only paired development experiment; never uses holdout questions."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

from natural_language_planner import NaturalLanguageResearchPlanner
from planner_acceptance import load_acceptance_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--cases', nargs='+', default=['pdac_known_targets', 'kras_variant_drugs'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--modes', nargs='+', choices=['off', 'synthetic_v1'],
                        default=['off', 'synthetic_v1'])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.num_ctx < 16384:
        parser.error('Comparison requires at least 16384 context tokens for full table, examples and generation')
    load_dotenv()
    cases = {c.case_id: c for c in load_acceptance_cases(Path(__file__).parent / 'tests/fixtures/planner_acceptance_cases.json')}
    selected = [cases[c] for c in args.cases]
    if any(c.evaluation_set != 'development' for c in selected):
        parser.error('Only development cases may be used')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['PLANNER_OPERATION_CATALOG_MODE'] = 'full'
    os.environ['OLLAMA_NUM_CTX'] = str(args.num_ctx)
    (output / 'experiment.json').write_text(json.dumps({
        'cases': args.cases, 'repeats': args.repeats,
        'planned_runs': len(selected) * args.repeats * len(args.modes),
        'num_ctx': args.num_ctx, 'catalog_mode': 'full', 'max_attempts': 1,
        'example_version': 'synthetic_v1', 'expert_approved': False,
        'note': 'Synthetic transfer examples; not proof of unseen-capability generalization.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for repeat in range(1, args.repeats + 1):
        for case in selected:
            modes = list(args.modes)
            if len(modes) > 1 and repeat % 2 == 0:
                modes.reverse()
            for mode in modes:
                os.environ['PLANNER_FEW_SHOT_MODE'] = mode
                name = f'{case.case_id}_r{repeat}_{mode}'
                print(f'START {name}', flush=True)
                report = NaturalLanguageResearchPlanner().run_with_report(
                    case.question, acceptance_case=case,
                    checkpoint_path=output / f'{name}.json', max_attempts=1)
                rows.append({'run': name, 'status': report.status,
                             'seconds': report.total_duration_seconds,
                             'first_attempt_passed': report.first_attempt_passed})
                (output / 'summary.json').write_text(json.dumps({
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'comparison': 'first_attempt_only_no_repair',
                    'expert_review': 'synthetic_examples_not_expert_approved',
                    'runs': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'END {name}: {report.status} ({report.total_duration_seconds:.1f}s)', flush=True)
                if report.attempts[-1].model_generation_timeout:
                    print('STOP: timeout; no next request while model cleanup may be incomplete.', flush=True)
                    return


if __name__ == '__main__':
    main()
