"""Generate artificial grades and random scores to exercise the evaluator, not a model benchmark."""
import csv
from pathlib import Path
import numpy as np


def main():
    output = Path('ats_benchmark/examples')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with (output / 'judgments.csv').open('w', newline='') as jf, (output / 'runs.csv').open('w', newline='') as rf:
        judgments, runs = csv.writer(jf), csv.writer(rf)
        judgments.writerow(['job_id', 'candidate_id', 'grade', 'split'])
        runs.writerow(['system', 'job_id', 'candidate_id', 'score'])
        for job in range(12):
            for candidate in range(15):
                grade = int(rng.integers(0, 4))
                jid, cid = f'synthetic-job-{job:02}', f'synthetic-candidate-{candidate:02}'
                judgments.writerow([jid, cid, grade, 'validation' if job < 4 else 'test'])
                for system in ['synthetic_baseline', 'synthetic_challenger']:
                    runs.writerow([system, jid, cid, float(rng.normal())])
    print(f'Wrote artificial evaluator fixtures to {output}; no ATS was tested.')


if __name__ == '__main__':
    main()
