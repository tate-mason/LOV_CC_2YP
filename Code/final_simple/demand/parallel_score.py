"""Evaluate independent household likelihood blocks on server CPU workers.

The simulated likelihood and its gradient are unchanged. Integration always
uses each household's entire history. No records are sent outside the process.
"""
from contextlib import contextmanager
import multiprocessing as mp
import numpy as np

_blocks = None
_score = None


def partition_households(data, workers):
    ranges = np.array_split(np.arange(data['n_households']), min(workers, data['n_households']))
    blocks = []
    rows = len(data['hh'])
    for ids in ranges:
        selected = np.flatnonzero((data['hh'] >= ids[0]) & (data['hh'] <= ids[-1]))
        block = {}
        for key, value in data.items():
            if key == 'labels':
                block[key] = value[ids]
            elif isinstance(value, np.ndarray) and value.ndim and len(value) == rows:
                block[key] = value[selected]
            else:
                block[key] = value
        block['hh'] = data['hh'][selected] - ids[0]
        block['n_households'] = len(ids)
        blocks.append(block)
    return blocks


def _evaluate(arguments):
    index, theta, draws, model = arguments
    block = _blocks[index]
    value, gradient = _score(theta, block, draws, model)
    return block['n_households'], value, gradient


@contextmanager
def parallel_objective(estimation, data, workers):
    """Linux fork shares read-only inputs; return only aggregate scores."""
    global _blocks, _score
    if workers <= 1:
        yield
        return
    if 'fork' not in mp.get_all_start_methods():
        raise RuntimeError('Parallel empirical score requires Linux fork on the research server.')
    _blocks = partition_households(data, workers)
    original = estimation.demand_objective_gradient
    _score = original
    # Launch before replacing the objective: workers retain the serial function.
    pool = mp.get_context('fork').Pool(len(_blocks))

    def objective(theta, candidate_data, draws, model, household=False):
        if household or candidate_data is not data:
            return original(theta, candidate_data, draws, model, household=household)
        results = pool.map(_evaluate, [(i, theta, draws, model) for i in range(len(_blocks))])
        total = sum(n for n, _, _ in results)
        return (sum(n * value for n, value, _ in results) / total,
                sum(n * gradient for n, _, gradient in results) / total)

    estimation.demand_objective_gradient = objective
    try:
        yield
    finally:
        estimation.demand_objective_gradient = original
        pool.close()
        pool.join()
        _blocks = None
        _score = None
