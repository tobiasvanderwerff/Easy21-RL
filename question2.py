"""
Question 2: Monte-Carlo control in Easy21
"""

import random
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from step import step
from util import State, Card, Action, CardColor, CARD_VALS, init_state
from mc import mc_control
from sarsa import sarsa_lambda


def main():
    n_episodes = 500000
    q_mc = mc_control(n_episodes=n_episodes)

    # print(f"State-action values after {n_episodes} episodes:")
    # for s, av in state_action_values.items():
    #     print(f"dealer card: {s.dealer_first_card.value()}, player sum: {s.player_sum}: {av}")

    # Save the maximum Q value for each state.
    max_action_val = np.zeros((len(CARD_VALS), 21))
    for s, av in q_mc.items():
        if s.is_terminal:
            continue
        dealer_first_card = s.dealer_first_card.value()
        player_sum = s.player_sum
        max_action_val[dealer_first_card-1, player_sum-1] = max(av.values())

    # Plot the action values in a 3D plot.
    fig = plt.figure(figsize=(18, 12))
    title = f"MC value function ({n_episodes} episodes)"
    x = np.arange(CARD_VALS[0], CARD_VALS[-1] + 1)  # dealer first card
    y = np.arange(1, 22)  # player sum
    xv, yv = np.meshgrid(x, y)
    ax = plt.axes(projection="3d")
    ax.plot_surface(xv, yv, max_action_val.transpose((1, 0)), cmap="viridis", edgecolor="green")
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer showing")
    ax.set_yticks(np.arange(21), labels=np.arange(1, 22))
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xticks(np.arange(len(CARD_VALS)), labels=np.arange(CARD_VALS[0], CARD_VALS[-1] + 1))
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_title(title)
    plt.waitforbuttonpress()

    # Plot the action values in a heatmap.
    plt.figure(figsize=(18, 12))
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
    plt.title(title)
    plt.tight_layout()
    plt.waitforbuttonpress()


if __name__ == "__main__":
    main()