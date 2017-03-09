"""
Various tests
"""

import numpy as np

from plots import get_dominant_strategy, get_domain_durations


def test_dominant_strategies():
    lattice = np.array([
        [0, 1, 1],
        [1, 2, 2]
    ])
    assert get_dominant_strategy(lattice) == 1
    assert (get_dominant_strategy(lattice, 2) == [1, 2]).all()
    assert (get_dominant_strategy(lattice, 3) == [1, 2, 0]).all()
    assert (get_dominant_strategy(lattice, 4) == [1, 2, 0, -1]).all()

    lattice = np.ones(shape=(50,50))*4
    lattice[5,5] = 5
    assert get_dominant_strategy(lattice, 1) == 4
    assert (get_dominant_strategy(lattice, 2) == [4, 5]).all()
    assert (get_dominant_strategy(lattice, 3) == [4, 5, -1]).all()

def test_domain_durations():
    series = [0, 0, 1, 2, 2, 2, 0]
    durations = get_domain_durations(series)
    assert (durations == [2, 1, 3, 1]).all()

    series = np.r_[np.ones(20), np.zeros(13), np.ones(42)*5]
    durations = get_domain_durations(series)
    assert (durations == [20, 13, 42]).all()
