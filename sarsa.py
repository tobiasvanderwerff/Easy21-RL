from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import random
from function_approx import state_to_features

from util import Action, State, init_state
from step import step 
from mc import epsilon_greedy_sample, get_epsilon

import numpy as np


def sarsa_lambda(n_episodes: int, lmbda: float, calculate_msa: bool = False, 
                 q_target: Optional[Dict[State, Dict[Action, float]]] = None
                 ) -> Tuple[Dict[State, Dict[Action, float]], Optional[List[float]]]:
    """
    Control using SARSA and TD-lambda.
    
    Args:
        n_episodes: number of episodes to run the algorithm
        lmbda: lambda parameter for SARSA-lambda
        calculate_msa: whether to calculate mean-squared error every episode. If True, `q_target` must be set.
        q_target: target q function, used for calculating mean-squared error
    Returns:
        - learned state-action function
        - msa per episode (optional)
    """
    state_to_action_counts = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})
    state_to_action_values = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})
    msa_per_episode = [0 for _ in range(n_episodes)] if calculate_msa else None

    for ep in range(n_episodes):
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
        # Calculate mean-squared error.
        if calculate_msa:
            msa = sum((state_to_action_values[s][a] - q_target[s][a]) ** 2 for s, av in q_target.items() for a, v in av.items()) / (len(q_target) * 2)
            msa_per_episode[ep] = msa
    return state_to_action_values, msa_per_episode


def sarsa_lambda_fapprox(n_episodes: int, lmbda: float, epsilon: float = 0.05, step_size: float = 0.01, n_features: int = 36,
                         calculate_msa: bool = False, q_target: Optional[Dict[State, Dict[Action, float]]] = None,
                        ) -> Tuple[np.ndarray, Optional[List[float]]]:
    """
    Control using SARSA-lambda and linear function approximation. Incremental updates are used to train
    the function approximator.
    
    Args:
        n_episodes: number of episodes to run the algorithm
        lmbda: lambda parameter for SARSA-lambda
        epsilon: epsilon for epsilon-greedy exploration
        step_size: step size (learning rate) for updating state-action values
        n_features: number of features in the feature vector used for linear function approximation
        calculate_msa: whether to calculate mean-squared error every episode. If True, `q_target` must be set.
        q_target: target q function, used for calculating mean-squared error
    Returns:
        - learned state-action function
        - msa per episode (optional)
    """
    msa_per_episode = [0 for _ in range(n_episodes)] if calculate_msa else None
    actions = [Action.HIT, Action.STICK]
    # Initialize weights to zero to make initial outputs and updates zero. 
    w = np.zeros((n_features,))  
    etrace = np.zeros((n_features,))

    for ep in range(n_episodes):
        s = init_state()
        # Select initial action.
        actions.sort(key=lambda a: w.T @ state_to_features(s, a), reverse=True)
        a = actions[0]
        while not s.is_terminal:
            # Take action.
            s_new, r = step(s, a)
            # Sample the next action.
            actions.sort(key=lambda a: w.T @ state_to_features(s, a), reverse=True)
            a_new = epsilon_greedy_sample(actions, epsilon)

            # Given quintuple (s, a, r, s_new, a_new), update the action-value function using backward-view TD-lambda.
            td_error = r + w.T @ state_to_features(s_new, a_new) - w.T @ state_to_features(s, a)
            etrace += state_to_features(s, a)
            w += step_size * td_error * etrace
            etrace *= lmbda

            s = s_new
            a = a_new
        # Calculate mean-squared error.
        if calculate_msa:
            msa = sum((w.T @ state_to_features(s, a) - q_target[s][a]) ** 2 for s, av in q_target.items() for a, v in av.items()) / (len(q_target) * 2)
            msa_per_episode[ep] = msa
    return w, msa_per_episode