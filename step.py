from dataclasses import dataclass, asdict
from typing import Tuple
from copy import copy

from util import State, Action, Card


def step(s: State, a: Action) -> Tuple[State, int]:
    """
    Sample a new state given the current state and action.

    Args:
        s: current state
        a: current action
    Returns:
        Tuple (s', r), containing the new state and the reward
    """
    if s.is_terminal:
        return s, 0

    s_new = copy(s.__dict__)  # update from the current state
    if a is Action.STICK:
        # Play out the dealer's cards and return the final reward and terminal state.
        dealer_sum = s.dealer_first_card.value()
        while 1 <= dealer_sum < 17:
            dealer_sum += Card.sample().value()
        if dealer_sum < s.player_sum or dealer_sum > 21:
            reward = 1  # win
        elif dealer_sum == s.player_sum:
            reward = 0  # draw
        else:
            reward = -1 # lose
        s_new["is_terminal"] = True
    elif a is Action.HIT:
        s_new["player_sum"] += Card.sample().value()
        reward = 0
        if s_new["player_sum"] > 21 or s_new["player_sum"] < 1:
            reward = -1  # lose
            s_new["is_terminal"] = True
    s_new = State(**s_new)
    return s_new, reward
