"""Repeatable local closed-loop benchmark with connection reuse and excluded warmup."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


async def measure(base, job_ids, concurrency, requests, warmup, mode):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base, timeout=30, limits=limits, trust_env=False) as client:
        async def send(index):
            start = time.perf_counter()
            try:
                response = await client.post('/screen', json={'job_id': job_ids[index % len(job_ids)], 'k': 10, 'mode': mode})
                response.raise_for_status()
                if len(response.json()['items']) != 10:
                    raise ValueError('wrong response size')
                return (time.perf_counter() - start) * 1000, None
            except Exception as error:
                return (time.perf_counter() - start) * 1000, type(error).__name__
        for index in range(warmup):
            _, error = await send(index)
            if error:
                raise RuntimeError(f'warmup failed: {error}')
        # Fixed workers keep issuing requests. Latency starts on dispatch and
        # excludes time waiting in a client task queue (closed-loop semantics).
        async def worker(worker_id):
            return [await send(index) for index in range(worker_id, requests, concurrency)]
        start = time.perf_counter()
        chunks = await asyncio.gather(*(worker(i) for i in range(concurrency)))
        elapsed = time.perf_counter() - start
        observations = [item for chunk in chunks for item in chunk]
        latency = [ms for ms, error in observations if error is None]
        errors = [error for _, error in observations if error is not None]
        return {'concurrency': concurrency, 'requests': requests, 'successes': len(latency),
                'errors': len(errors), 'error_types': sorted(set(errors)), 'elapsed_s': elapsed,
                'successful_requests_per_second': len(latency) / elapsed,
                'latency_ms_successful': {f'p{p}': float(np.percentile(latency, p)) if latency else None
                                          for p in [50, 95, 99]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 8, 20])
    parser.add_argument('--mode', choices=['bm25', 'neural', 'hybrid'], default='hybrid')
    parser.add_argument('--port', type=int, default=8124)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/load_results.json')
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or any(c < 1 or c > args.requests for c in args.concurrency):
        parser.error('positive request counts and concurrency <= requests required')
    job_ids = json.loads((ROOT / 'data/split.json').read_text())['test_job_ids'][::12][:10]
    base = f'http://127.0.0.1:{args.port}'
    with tempfile.TemporaryFile(mode='w+') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'serving.app:app',
                                 '--host', '127.0.0.1', '--port', str(args.port), '--workers', '1'],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                deadline = time.monotonic() + 45
                while True:
                    if proc.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        if client.get(base + '/health').status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('server startup exceeded 45 seconds')
                    time.sleep(0.2)
            results = [asyncio.run(measure(base, job_ids, c, args.requests, args.warmup, args.mode))
                       for c in args.concurrency]
            report = {'mode': args.mode, 'workers': 1, 'warmup_excluded_per_scenario': args.warmup,
                      'method': 'closed-loop local HTTP, reused connections, dispatch-to-response latency',
                      'python': platform.python_version(), 'platform': platform.platform(),
                      'logical_cpus_visible': os.cpu_count(), 'results': results,
                      'limitations': ['Client and server share a host; not a production capacity estimate.',
                                      'Fixed request count, not a long-running soak test.',
                                      'No cross-version speedup claim; old test created a client per request.']}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    main()
