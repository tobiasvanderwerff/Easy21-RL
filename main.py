import random

from step import step
from util import State, Card, Action, CardColor, CARD_VALS, init_state
from mc import mc_control


n_episodes = 100000

state_action_values = mc_control(n_episodes)

print(f"State-action values after {n_episodes} episodes:")
for s, av in state_action_values.items():
    print(f"{s}: {av}")

# total_reward = 0
# for _ in range(n_episodes):
#     s = init_state() 
#     while not s.is_terminal:
#         a = Action.HIT if s.player_sum < 17 else Action.STICK  # simple policy
#         s, r = step(s, a)
#     total_reward += r

# print(f"Total reward after {n_episodes} episodes is {total_reward}.")
