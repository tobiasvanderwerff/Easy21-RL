from util import State, Action

import numpy as np


def state_to_features(s: State, a: Action) -> np.ndarray:
    """Map a (state, action) pair to its corresponding feature vector."""
    n1, n2, n3 = 3, 6, 2  # number of intervals per attribute
    f1 = np.zeros((n1,))
    f2 = np.zeros((n2,))
    f3 = np.zeros((n3,))
    res = np.zeros((n1 * n2 * n3,))

    # Register the cuboids that the (s, a) tuple lies in (note that they can overlap).
    if s.dealer_first_card in range(1, 5):
        f1[0] = 1
    if s.dealer_first_card in range(4, 8):
        f1[1] = 1
    if s.dealer_first_card in range(7, 11):
        f1[2] = 1

    if s.player_sum in range(1, 7):
        f2[0] = 1
    if s.player_sum in range(4, 10):
        f2[1] = 1
    if s.player_sum in range(7, 13):
        f2[2] = 1
    if s.player_sum in range(10, 16):
        f2[3] = 1
    if s.player_sum in range(13, 19):
        f2[4] = 1
    if s.player_sum in range(16, 22):
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
