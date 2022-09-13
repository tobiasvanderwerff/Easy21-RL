import random

import matplotlib.pyplot as plt
import numpy as np

from step import step
from util import State, Card, Action, CardColor, CARD_VALS, init_state
from mc import mc_control


n_episodes = 500000

state_action_values = mc_control(n_episodes)

# print(f"State-action values after {n_episodes} episodes:")
# for s, av in state_action_values.items():
#     print(f"dealer card: {s.dealer_first_card.value()}, player sum: {s.player_sum}: {av}")

max_action_val = np.zeros((len(CARD_VALS), 21))
for s, av in state_action_values.items():
    dealer_first_card = s.dealer_first_card.value()
    player_sum = s.player_sum
    max_action_val[dealer_first_card-1, player_sum-1] = max(av.values())

# Plot the action values.
fig = plt.figure(figsize=(18, 12))
# x = np.arange(CARD_VALS[0], CARD_VALS[-1] + 1)  # dealer first card
# y = np.arange(1, 22)  # player sum
# xv, yv = np.meshgrid(x, y)
# ax1 = fig.add_subplot(1, 1, 1, projection="3d")
# ax1.plot_surface(xv, yv, max_action_val)

plt.imshow(max_action_val)
for i in range(len(CARD_VALS)):
    for j in range(21):
        plt.text(j, i, round(max_action_val[i, j], 2))

plt.xlabel("Player sum")
plt.ylabel("Dealer showing")
plt.xticks(np.arange(21), labels=np.arange(1, 22))
plt.xticks([0, 5, 10, 15, 20])
plt.yticks(np.arange(len(CARD_VALS)), labels=np.arange(CARD_VALS[0], CARD_VALS[-1] + 1))
plt.yticks([0, 2, 4, 6, 8])
plt.title(f"MC value function ({n_episodes} episodes)")
plt.tight_layout()
plt.show()
