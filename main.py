import random

import matplotlib.pyplot as plt
import numpy as np

from step import step
from util import State, Card, Action, CardColor, CARD_VALS, init_state
from mc import mc_control
from sarsa import sarsa_lambda


def main(lmbda: float = 0, use_mc: bool = True):
    """
    Args:
        lmbda: lambda parameter for SARSA-lambda
        use_mc: whether to use Monte-Carlo control. Otherwise, use SARSA-lambda.
    """
    if use_mc:
        n_episodes = 500000
        state_action_values = mc_control(n_episodes)
    else:
        n_episodes = 1000
        state_action_values = sarsa_lambda(n_episodes, lmbda=0)

    # print(f"State-action values after {n_episodes} episodes:")
    # for s, av in state_action_values.items():
    #     print(f"dealer card: {s.dealer_first_card.value()}, player sum: {s.player_sum}: {av}")

    max_action_val = np.zeros((len(CARD_VALS), 21))
    for s, av in state_action_values.items():
        if s.is_terminal:
            continue
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

    title = f"MC value function ({n_episodes} episodes)" if use_mc else f"SARSA({lmbda}) value function ({n_episodes} episodes)"
    plt.xlabel("Player sum")
    plt.ylabel("Dealer showing")
    plt.xticks(np.arange(21), labels=np.arange(1, 22))
    plt.xticks([0, 5, 10, 15, 20])
    plt.yticks(np.arange(len(CARD_VALS)), labels=np.arange(CARD_VALS[0], CARD_VALS[-1] + 1))
    plt.yticks([0, 2, 4, 6, 8])
    plt.title(title)
    plt.tight_layout()
    plt.waitforbuttonpress()
    # plt.show()


if __name__ == "__main__":
    lmbda = 1.0
    use_mc = False

    main(lmbda, use_mc)