from functools import cache

from util import State, Action

import numpy as np


def fapprox_action_argmax(s: State, w: np.ndarray) -> Action:
    """
    Return the argmax over action values, where state-action values are determined 
    using a linear function approximator.
    """
    v1 = w.T @ state_to_features(s, Action.HIT)
    v2 = w.T @ state_to_features(s, Action.STICK)
    return Action.HIT if v1 > v2 else Action.STICK


@cache
def state_to_features(s: State, a: Action) -> np.ndarray:
    """Map a (state, action) pair to its corresponding feature vector."""
    n1, n2, n3 = 3, 6, 2  # number of intervals per attribute
    f1 = np.zeros((n1,))
    f2 = np.zeros((n2,))
    f3 = np.zeros((n3,))
    res = np.zeros((n1 * n2 * n3,))

    # Register the cuboids that the (s, a) tuple lies in (note that they can overlap).
    if 1 <= s.dealer_first_card.val <= 4:
        f1[0] = 1
    if 4 <= s.dealer_first_card.val <= 7:
        f1[1] = 1
    if 7 <= s.dealer_first_card.val <= 10:
        f1[2] = 1

    if 1 <= s.player_sum <= 6:
        f2[0] = 1
    if 4 <= s.player_sum <= 9:
        f2[1] = 1
    if 7 <= s.player_sum <= 12:
        f2[2] = 1
    if 10 <= s.player_sum <= 15:
        f2[3] = 1
    if 13 <= s.player_sum <= 18:
        f2[4] = 1
    if 16 <= s.player_sum <= 21:
        f2[5] = 1

    if a is Action.HIT:
        f3[0] = 1
    else:
        f3[1] = 1

    # Create the feature vector.
    for i1 in np.flatnonzero(f1):
        for i2 in np.flatnonzero(f2):
            for i3 in np.flatnonzero(f3):
                res[i1 * n2 * n3 + i2 * n3 + i3] = 1
    return res
