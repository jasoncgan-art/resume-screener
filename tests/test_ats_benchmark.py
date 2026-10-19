import csv
import pytest
from ats_benchmark.evaluate import ranking_metrics, compare, load_inputs


def test_known_ranking_and_missing_relevance():
    grades = {'a': 3, 'b': 2, 'c': 0}
    perfect = ranking_metrics(grades, {'a': 3, 'b': 2, 'c': 1}, 2)
    assert perfect['ndcg'] == 1
    assert perfect['precision'] == 1
    assert perfect['recall_within_judged_pool'] == 1
    assert ranking_metrics({'a': 0}, {'a': 1})['ndcg'] is None
    with pytest.raises(ValueError):
        ranking_metrics(grades, {'a': 1})


def test_paired_identical_systems_have_zero_difference():
    labels = {'j1': {'a': 3, 'b': 0}, 'j2': {'a': 0, 'b': 3}}
    run = {'j1': {'a': 1, 'b': 0}, 'j2': {'a': 1, 'b': 0}}
    report = compare(labels, {'base': run, 'new': run}, 'base', draws=100)
    assert report['comparisons']['new']['paired_job_bootstrap_95_ci'] == [0, 0]


def test_one_job_does_not_produce_false_certainty():
    labels = {'j': {'a': 3, 'b': 0}}
    report = compare(labels, {'base': {'j': {'a': 0, 'b': 1}},
                              'new': {'j': {'a': 1, 'b': 0}}}, 'base')
    assert report['comparisons']['new']['paired_job_bootstrap_95_ci'] is None


def test_loader_rejects_unjudged_or_mismatched_pools(tmp_path):
    judgment = tmp_path / 'judgments.csv'
    runs = tmp_path / 'runs.csv'
    judgment.write_text('job_id,candidate_id,grade,split\nj,a,3,test\nj,b,0,test\n')
    runs.write_text('system,job_id,candidate_id,score\nbase,j,a,1\nbase,j,b,0\nnew,j,a,1\n')
    with pytest.raises(ValueError, match='universe'):
        load_inputs(judgment, runs)


def test_loader_rejects_split_leakage(tmp_path):
    judgment = tmp_path / 'judgments.csv'
    runs = tmp_path / 'runs.csv'
    judgment.write_text('job_id,candidate_id,grade,split\nj,a,3,test\nj,b,0,train\n')
    runs.write_text('system,job_id,candidate_id,score\nbase,j,a,1\nnew,j,a,1\n')
    with pytest.raises(ValueError, match='split'):
        load_inputs(judgment, runs)
