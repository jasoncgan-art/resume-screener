"""Metric sanity checks on hand-computed examples."""

import math

import numpy as np

from evaluate import rank_metrics


def test_perfect_ranking():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    m = rank_metrics(scores, {0})
    assert m["recall@10"] == 1.0
    assert m["recall@50"] == 1.0
    assert m["ndcg@10"] == 1.0
    assert m["mrr"] == 1.0


def test_second_place():
    # relevant doc at rank 1 (0-indexed): MRR = 1/2, NDCG = (1/log2(3))/(1/log2(2))
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    m = rank_metrics(scores, {2})
    assert m["mrr"] == 0.5
    assert m["ndcg@10"] == math.log2(2) / math.log2(3)
    assert m["recall@10"] == 1.0


def test_no_relevant_retrieved_in_topk():
    scores = np.array([0.9, 0.8, 0.7] + [0.0] * 20)
    m = rank_metrics(scores, {22})  # relevant sits at rank 22
    assert m["recall@10"] == 0.0
    assert m["ndcg@10"] == 0.0
    assert m["mrr"] == 1.0 / 23


def test_tie_break_deterministic():
    # equal scores -> lower doc index first (stable, documented tie-break)
    scores = np.array([0.5, 0.5, 0.5])
    m = rank_metrics(scores, {2})
    assert m["mrr"] == 1.0 / 3


def test_empty_relevant_set():
    m = rank_metrics(np.array([0.9, 0.1]), set())
    assert m["recall@10"] == 0.0
    assert m["mrr"] == 0.0
