"""Strict, paired ranking comparison. No resumes or protected attributes needed."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path, fields):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != set(fields):
            raise ValueError(f'{path}: expected exactly {fields}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty file')
    if any(any(value is None or not value.strip() for value in row.values()) for row in rows):
        raise ValueError(f'{path}: blank fields are not judgments')
    return rows


def load_inputs(judgments_path, runs_path, split='test'):
    judgments = read_rows(judgments_path, ['job_id', 'candidate_id', 'grade', 'split'])
    runs = read_rows(runs_path, ['system', 'job_id', 'candidate_id', 'score'])
    labels, job_split, seen = defaultdict(dict), {}, set()
    for row in judgments:
        job, candidate = row['job_id'], row['candidate_id']
        if (job, candidate) in seen:
            raise ValueError('duplicate judgment; adjudicate reviewers before evaluation')
        seen.add((job, candidate))
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('unknown split')
        if job in job_split and job_split[job] != row['split']:
            raise ValueError('job appears in more than one split')
        job_split[job] = row['split']
        grade = int(row['grade'])
        if grade not in (0, 1, 2, 3):
            raise ValueError('grade must be an integer 0..3')
        if row['split'] == split:
            labels[job][candidate] = grade
    if not labels:
        raise ValueError('no judgments in requested split')
    systems, run_seen = defaultdict(lambda: defaultdict(dict)), set()
    for row in runs:
        system, job, candidate = row['system'], row['job_id'], row['candidate_id']
        if job not in job_split:
            raise ValueError('run contains a job with no judgments')
        key = (system, job, candidate)
        if key in run_seen:
            raise ValueError('duplicate candidate in a system run')
        run_seen.add(key)
        score = float(row['score'])
        if not math.isfinite(score):
            raise ValueError('scores must be finite')
        if job in labels:
            systems[system][job][candidate] = score
    if len(systems) < 2:
        raise ValueError('comparison requires at least two systems')
    for system, jobs in systems.items():
        if set(jobs) != set(labels):
            raise ValueError(f'{system}: missing jobs; cannot compare different job samples')
        for job, scores in jobs.items():
            if set(scores) != set(labels[job]):
                raise ValueError(f'{system}/{job}: candidate universe mismatch or unjudged candidates')
    return dict(labels), {s: dict(j) for s, j in systems.items()}


def ranking_metrics(grades, scores, k=10):
    if k < 1 or set(grades) != set(scores) or not grades:
        raise ValueError('invalid k or candidate universe')
    # Stable ID tie-break, identical across systems; never use relevance to break ties.
    order = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
    cutoff = min(k, len(order))
    top = [grades[candidate] for candidate in order[:cutoff]]
    ideal = sorted(grades.values(), reverse=True)[:cutoff]
    def dcg(values):
        return sum((2 ** grade - 1) / math.log2(rank + 2)
                   for rank, grade in enumerate(values))
    denominator = dcg(ideal)
    positives = sum(grade >= 2 for grade in grades.values())
    hits = sum(grade >= 2 for grade in top)
    return {'ndcg': dcg(top) / denominator if denominator else None,
            'precision': hits / cutoff,
            'recall_within_judged_pool': hits / positives if positives else None,
            'evaluated_cutoff': cutoff, 'judged_candidates': len(grades)}


def paired_interval(deltas, seed=42, draws=2000):
    """Resample jobs, not candidate rows. One job cannot estimate between-job uncertainty."""
    if len(deltas) < 2:
        return None
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    means = [float(rng.choice(deltas, len(deltas), replace=True).mean()) for _ in range(draws)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare(labels, systems, baseline, k=10, seed=42, draws=2000):
    if baseline not in systems:
        raise ValueError('baseline system is absent')
    per_job = {system: {job: ranking_metrics(labels[job], scores, k)
                        for job, scores in jobs.items()} for system, jobs in systems.items()}
    metric_names = ('ndcg', 'precision', 'recall_within_judged_pool')
    aggregates = {}
    for system, jobs in per_job.items():
        aggregates[system] = {}
        for metric in metric_names:
            values = [row[metric] for row in jobs.values() if row[metric] is not None]
            aggregates[system][metric] = float(np.mean(values)) if values else None
    comparisons = {}
    for system, jobs in per_job.items():
        if system == baseline:
            continue
        eligible = sorted(job for job in jobs if jobs[job]['ndcg'] is not None)
        deltas = [jobs[job]['ndcg'] - per_job[baseline][job]['ndcg'] for job in eligible]
        interval = paired_interval(deltas, seed, draws)
        comparisons[system] = {
            'mean_ndcg_delta': float(np.mean(deltas)) if deltas else None,
            'paired_job_bootstrap_95_ci': interval, 'eligible_jobs': len(deltas),
            'jobs_improved': sum(d > 0 for d in deltas),
            'jobs_worsened': sum(d < 0 for d in deltas),
            'interpretation': ('positive interval on this judged sample only'
                               if interval and interval[0] > 0 else
                               'no clear positive NDCG difference established')}
    return {'baseline': baseline, 'k': k, 'seed': seed, 'bootstrap_draws': draws,
            'n_jobs': len(labels), 'metrics': aggregates, 'comparisons': comparisons,
            'per_job': per_job,
            'limitations': [
                'Recall is only within the supplied fully judged candidate pool, not all applicants.',
                'No score is a probability of candidate suitability or a hiring decision.',
                'Bootstrap assumes independent jobs; near-duplicate requisitions violate this assumption.',
                'Reviewer agreement, subgroup effects and business impact need separate validation.',
                'Do not repeatedly tune on this test report; preserve a new holdout for the final study.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judgments', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--data-kind', choices=['synthetic', 'human-reviewed'], required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    labels, systems = load_inputs(args.judgments, args.runs, args.split)
    result = compare(labels, systems, args.baseline, args.k)
    result.update({'data_kind': args.data_kind, 'split': args.split,
                   'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for name, path in [('judgments', args.judgments), ('runs', args.runs)]}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('data_kind', 'metrics', 'comparisons')}, indent=2))


if __name__ == '__main__':
    main()
