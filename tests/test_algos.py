import numpy as np

from offshore_methane.algos import instances_from_probs


def test_instances_from_probs_positive():
    arr = np.array([[0.8, 0.6], [0.7, 0.2]], dtype=float)
    feats = instances_from_probs(arr, 0.5, 0.5, 0.5)
    assert len(feats) == 1
    mean_conf = feats[0].properties['mean_conf']
    assert np.isclose(mean_conf, np.mean([0.8, 0.6, 0.7]))


def test_instances_from_probs_negative():
    arr = np.array([[-0.6, -0.6], [-0.6, 0.0]], dtype=float)
    feats = instances_from_probs(arr, -0.5, -0.5, -0.5)
    assert len(feats) == 1
    assert feats[0].properties['max_conf'] == -0.6

