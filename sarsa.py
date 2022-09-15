from collections import defaultdict, deque
from typing import Dict
import random

from util import Action, State, init_state
from step import step 
from mc import epsilon_greedy_sample, get_epsilon

import numpy as np


def sarsa_lambda(n_episodes: int, lmbda: float) -> Dict[State, Dict[Action, float]]:
    """Control using SARSA and TD-lambda."""
    state_to_action_counts = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})
    state_to_action_values = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})

    for _ in range(n_episodes):
        state_to_action_etrace = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})  # eligibility traces
        s = init_state()
        # Select initial action.
        action_values = state_to_action_values[s]
        actions = list(action_values.keys())
        actions.sort(key=action_values.get, reverse=True)
        a = actions[0]
        while not s.is_terminal:
            # Take action.
            s_new, r = step(s, a)
            # Sample the next action.
            action_values = state_to_action_values[s]
            actions.sort(key=action_values.get, reverse=True)
            N_s = sum(ct for ct in state_to_action_counts[s].values())
            epsilon = get_epsilon(N_s)
            a_new = epsilon_greedy_sample(actions, epsilon)

            # Given quintuple (s, a, r, s_new, a_new), update the action-value function using backward-view TD-lambda.
            td_error = r + state_to_action_values[s_new][a_new] - state_to_action_values[s][a]
            state_to_action_etrace[s][a] += 1  # update eligibility trace for visited state
            state_to_action_counts[s][a] += 1  # increment visit count
            # TODO: could use vector operations to avoid the for-loops.
            for s_, av in state_to_action_values.items():
                for a_, val in av.items():
                    if state_to_action_counts[s_][a_] == 0 or state_to_action_etrace[s_][a_] == 0:
                        continue
                    step_size = 1 / state_to_action_counts[s_][a_]
                    etrace = state_to_action_etrace[s_][a_]
                    state_to_action_values[s_][a_] += step_size * td_error * etrace   # TD update
                    state_to_action_etrace[s_][a_] *= lmbda  # update eligibility trace
            s = s_new
            a = a_new
    return state_to_action_values
