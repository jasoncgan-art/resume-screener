"""Rank an approved local text corpus; export scores for ats_benchmark.

No network calls, ATS writes, automatic rejection or resume parsing. JSON text
must already be extracted and reviewed. Neural weights were trained on synthetic
data and are experimental on other domains.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from dataset import encode, pad_batch, tokenize
from evaluate import BM25Index
from hybrid import fuse_scores
from models import BiEncoder


def read_documents(path, id_key):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected a nonempty JSON list')
    seen = set()
    for row in rows:
        if set(row) != {id_key, 'text'}:
            raise ValueError(f'expected exactly {id_key} and text')
        if not isinstance(row[id_key], str) or not row[id_key].strip() or row[id_key] in seen:
            raise ValueError('IDs must be unique nonempty strings')
        if not isinstance(row['text'], str) or not tokenize(row['text']):
            raise ValueError('empty or untokenizable document')
        seen.add(row[id_key])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--system', default='prototype')
    parser.add_argument('--mode', choices=['bm25', 'neural', 'hybrid'], default='bm25')
    parser.add_argument('--neural-weight', type=float)
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).parent / 'artifacts')
    args = parser.parse_args()
    if args.mode == 'hybrid' and (args.neural_weight is None or not 0 <= args.neural_weight <= 1):
        parser.error('hybrid requires --neural-weight in [0,1], chosen on validation jobs')
    torch.set_num_threads(2)
    jobs = read_documents(args.jobs, 'job_id')
    candidates = read_documents(args.candidates, 'candidate_id')
    index = BM25Index([tokenize(row['text']) for row in candidates])
    model, data, vectors = None, None, None
    if args.mode != 'bm25':
        data = torch.load(args.artifacts / 'dataset.pt', weights_only=False)
        artifact = torch.load(args.artifacts / 'biencoder.pt', weights_only=True)
        model = BiEncoder(data['vocab_size'], artifact['emb_dim'], artifact['hidden_dim'],
                          artifact['out_dim'], artifact['temperature'], n_skills=artifact['n_skills'])
        model.load_state_dict(artifact['state_dict'])
        model.eval()
        with torch.no_grad():
            vectors = torch.cat([
                model.encode_resumes(pad_batch([encode(row['text'], data['vocab'])
                                                for row in candidates[start:start + 128]]))
                for start in range(0, len(candidates), 128)]).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['system', 'job_id', 'candidate_id', 'score'])
        for job in jobs:
            lexical = index.scores(tokenize(job['text']))
            scores = lexical
            if model is not None:
                with torch.no_grad():
                    query = model.encode_jobs(pad_batch([encode(job['text'], data['vocab'])])).numpy()[0]
                semantic = vectors @ query
                scores = semantic if args.mode == 'neural' else fuse_scores(lexical, semantic, args.neural_weight)
            for row, score in zip(candidates, scores):
                writer.writerow([args.system, job['job_id'], row['candidate_id'], float(score)])
    manifest = {'mode': args.mode, 'neural_weight': args.neural_weight,
                'jobs': len(jobs), 'candidates_per_job': len(candidates),
                'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in [('jobs', args.jobs), ('candidates', args.candidates)]},
                'scope': 'offline decision-support prototype; no ATS comparison or hiring decision',
                'neural_limitations': 'synthetic-trained vocabulary; OOV and 128-token truncation require review'}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(jobs) * len(candidates)} scores to {args.out}')


if __name__ == '__main__':
    main()
