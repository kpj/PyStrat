"""
Various tests
"""

import numpy as np

from plots import get_dominant_strategy, get_domain_durations


def test_dominant_strategies():
    lattice = np.array([
        [0, 1],
        [1, 2]
    ])
    result = get_dominant_strategy(lattice)
    assert result == 1

    lattice = np.ones(shape=(50,50))*4
    lattice[5,5] = 5
    result = get_dominant_strategy(lattice)
    assert result == 4

def test_domain_durations():
    series = [0, 0, 1, 2, 2, 2, 0]
    durations = get_domain_durations(series)
    assert (durations == [2, 1, 3, 1]).all()

    series = np.r_[np.ones(20), np.zeros(13), np.ones(42)*5]
    durations = get_domain_durations(series)
    assert (durations == [20, 13, 42]).all()
