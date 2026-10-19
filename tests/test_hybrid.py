import numpy as np
import pytest
import torch
from hybrid import fuse_scores, stable_order, select_weight
from models import TextTower
from tests.test_train_smoke import _tiny_world


def test_rrf_endpoints_and_deterministic_ties():
    lexical, neural = [1, 3, 2], [3, 1, 2]
    assert stable_order(fuse_scores(lexical, neural, 0)).tolist() == [1, 2, 0]
    assert stable_order(fuse_scores(lexical, neural, 1)).tolist() == [0, 2, 1]
    assert stable_order([1, 1, 1]).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        fuse_scores([1], [1, 2], 0.5)
    with pytest.raises(ValueError):
        stable_order([np.nan])


def test_selector_can_choose_baseline_instead_of_forcing_neural_win():
    rows = [(0, np.array([3, 2, 1]), np.array([1, 2, 3]), {0})]
    weight, _ = select_weight(rows)
    assert weight == 0


def test_three_way_split_is_disjoint(tmp_path, monkeypatch):
    data = _tiny_world(tmp_path, monkeypatch)
    parts = [set(data[k]) for k in ('train_job_ids', 'validation_job_ids', 'test_job_ids')]
    assert all(parts)
    assert all(not parts[i] & parts[j] for i in range(3) for j in range(i))
    assert set.union(*parts) == set(range(len(data['jobs'])))


def test_empty_text_fails_instead_of_returning_nan_embeddings():
    model = TextTower(10)
    with pytest.raises(ValueError, match='padding'):
        model(torch.zeros((1, 3), dtype=torch.long))


def test_vocab_ties_do_not_depend_on_document_insertion_order():
    from dataset import build_vocab
    documents = ['python sql', 'sql python', 'java scala', 'scala java']
    assert build_vocab(documents) == build_vocab(list(reversed(documents)))
