from collections import defaultdict, deque
from typing import Dict, Sequence, List

from util import Action, State, init_state
from step import step

import numpy as np
from tqdm import tqdm


rng = np.random.default_rng()


def mc_control(n_episodes: int = 1000) -> Dict[State, Dict[Action, float]]:
    """
    Apply generalized policy iteration using Monte-Carlo evaluation and
    epsilon-greedy learning.

    Generalized policy evaluation looks like this:
    ```
    Repeat every episode:
        1. Policy evaluation: Evaluate the value function v_pi (in this case, using MC evaluation)
        2. Policy improvement: Improve the policy pi by choosing greedily w.r.t. the value function v_pi
    ```
    
    Because we use model-free reinforcement learning, we do not have access to the
    underlying MDP and therefore cannot integrate over state transition probabilities to
    obtain the state-value function V(s). Therefore, we use the action-value function
    Q(s, a) instead. 

    For MC evaluation, we use first-visit evaluation.

    Args:
        n_episodes: number of episodes to run MC control
    Returns:
        state-action function Q(s, a)
    """
    state_to_action_counts = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})
    state_to_action_values = defaultdict(lambda: {Action.HIT: 0, Action.STICK: 0})

    for _ in tqdm(range(n_episodes), desc="MC learning"):
        seen = defaultdict(lambda: {Action.HIT: False, Action.STICK: False})
        history = deque()  # history containing tuples (s_t, a_t, r_{t+1})
        s = init_state() 
        # Run a full episode.
        while not s.is_terminal:
            # Sample an action and take a step.
            action_values = state_to_action_values[s]
            actions = list(action_values.keys())
            actions.sort(key=action_values.get, reverse=True)
            # TODO (optimization): Could make the list of action-value pairs a priority queue for O(1) argmax operation
            # a_argmax = list(sorted(action_values.items(), key=lambda x: x[0]))[-1][0]  # argmax over action values
            N_s = sum(ct for ct in state_to_action_counts[s].values())
            epsilon = get_epsilon(N_s)
            a = epsilon_greedy_sample(actions, epsilon)
            s_new, r = step(s, a)
            history.append((s, a, r))
            s = s_new
        # Policy evaluation: Update state-action function.
        final_reward = history[-1][-1]  # reward is only obtained at the final step
        for s, a, r in history:
            if not seen[s][a] and not s.is_terminal:  # first-visit evaluation
                state_to_action_counts[s][a] += 1  # increment visit count
                step_size = 1 / state_to_action_counts[s][a]
                q = state_to_action_values[s][a]
                state_to_action_values[s][a] += step_size * (final_reward - q)
                seen[s][a] = True
    return state_to_action_values


def epsilon_greedy_sample(actions: List[Action], epsilon: float) -> Action:
    """Sample an action using epsilon-greedy. The first action should be the greedily sampled argmax action."""
    n_actions = len(actions)
    probs = [epsilon / n_actions for _ in range(n_actions)]
    probs[0] += 1 - epsilon
    sampled_action_idx = rng.multinomial(1, probs).argmax()  # categorical distribution, i.e. die throw
    return actions[sampled_action_idx]


def get_epsilon(N_s: int, N_0: int = 100) -> float:
    """Get (decaying) epsilon for epsilon-greedy exploration."""
    return N_0 / (N_0 + N_s)
