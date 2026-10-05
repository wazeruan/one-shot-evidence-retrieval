import json
import math

import pytest

from one_shot_evidence_retrieval.graphs import chart, running_mean, write_graphs


def record(query_id, recall=0.5, latency=12.0):
    return {
        'query_id': query_id,
        'metrics': {
            'baseline': {'support_recall': recall, 'complete_support': 0.0},
            'judged': {'support_recall': 1.0, 'complete_support': 1.0},
        },
        'timing_ms': {'baseline_online': latency, 'judge_extra': 24.0},
        'bundles': [
            {'combined': .1, 'coverage': {'score': .1, 'yes_no_mass': .2},
             'correctness': {'score': .2, 'yes_no_mass': .3}},
            {'combined': .9, 'coverage': {'score': .9, 'yes_no_mass': .8},
             'correctness': {'score': .7, 'yes_no_mass': .6}},
        ],
    }


def test_running_means_are_prefix_means():
    assert running_mean([]) == []
    assert running_mean([1, 0, .5, 1]) == pytest.approx([1, .5, .5, .625])


def test_single_point_is_visible_and_labels_escaped():
    svg = chart('<script>', [('A & B', [.5], '#123456')], bounds=(0, 1))
    assert '<circle ' in svg
    assert '65.00,115.00' in svg
    assert '<script>' not in svg
    assert '&lt;script&gt;' in svg
    assert 'A &amp; B' in svg


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_chart_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match='finite'):
        chart('bad', [('bad', [value], '#000')])


def test_chart_rejects_degenerate_bounds():
    with pytest.raises(ValueError, match='bounds'):
        chart('bad', [], bounds=(1, 1))


def test_progress_refresh_includes_all_records_and_winning_bundle(tmp_path):
    records = [record('q1')]
    write_graphs(tmp_path, records, 3)
    page = (tmp_path / 'progress.html').read_text()
    assert 'Completed 1 / 3 queries' in page
    assert 'Evaluation curves, not training losses' in page
    # Winning bundle's .9 coverage appears at y=43, rather than loser's .1.
    assert '65.00,43.00' in page
    records.append(record('q2', recall=1.0))
    write_graphs(tmp_path, records, 3)
    payload = json.loads((tmp_path / 'progress.json').read_text())
    assert payload['completed'] == 2
    assert payload['total'] == 3
    assert payload['query_ids'] == ['q1', 'q2']
    assert payload['metrics'][1]['timing_ms']['baseline_online'] == 12.0
    assert 'Completed 2 / 3 queries' in (tmp_path / 'progress.html').read_text()
    assert not list(tmp_path.glob('.*.tmp'))


def test_empty_records_do_not_create_misleading_graphs(tmp_path):
    write_graphs(tmp_path, [], 3)
    assert not list(tmp_path.iterdir())


def test_nonfinite_measured_latency_is_rejected_before_publish(tmp_path):
    with pytest.raises(ValueError, match='finite'):
        write_graphs(tmp_path, [record('q1', latency=math.nan)], 1)
    assert not (tmp_path / 'progress.html').exists()
