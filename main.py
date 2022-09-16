import random
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from step import step
from util import State, Card, Action, CardColor, CARD_VALS, init_state
from mc import mc_control
from sarsa import sarsa_lambda


def main(lmbda: float = 0):
    """
    Args:
        lmbda: lambda parameter for SARSA-lambda
    """
    print("Running MC control.")
    q_mc = mc_control(n_episodes=500000)

    print("Running SARSA-lambda control.")
    lmbdas = np.arange(0, 1.1, 0.1)
    msas = {lmb: 0 for lmb in lmbdas}
    lambda_to_msa_per_episode = {0.0: None, 1.0: None}
    for lmbda in lmbdas:
        q_sarsa, msas_ = sarsa_lambda(n_episodes=1000, lmbda=lmbda, calculate_msa=True, q_target=q_mc)
        if lmbda in [0.0, 1.0]:
            # Save msa for every episode for lambda=0 and lambda=1.
            lambda_to_msa_per_episode[lmbda] = msas_
        msas[lmbda] = msas_[-1]

    # Plot SARSA-lambda mean-squared error compared to MC value function.
    # plt.figure()
    # plt.plot(list(msas.keys()), list(msas.values()))
    # plt.xlabel("lambda")
    # plt.ylabel("mean-squared error")
    # plt.title("TD-lambda performance as a function of lambda")
    # plt.tight_layout()
    # plt.waitforbuttonpress()
    # return

    # Plot the learning curve of mean-squared error against episode number for SARSA(0) and SARSA(1).
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(len(lambda_to_msa_per_episode[0.0]))), list(lambda_to_msa_per_episode[0.0]),
             label="SARSA(0)")
    plt.plot(list(range(len(lambda_to_msa_per_episode[1.0]))), list(lambda_to_msa_per_episode[1.0]),
             label="SARSA(1)")
    plt.xlabel("episode")
    plt.ylabel("mean-squared error")
    plt.title(r"Learning curve for SARSA($\lambda$)")
    plt.legend()
    plt.tight_layout()
    plt.waitforbuttonpress()
    return

    # print(f"State-action values after {n_episodes} episodes:")
    # for s, av in state_action_values.items():
    #     print(f"dealer card: {s.dealer_first_card.value()}, player sum: {s.player_sum}: {av}")

    max_action_val = np.zeros((len(CARD_VALS), 21))
    for s, av in q_mc.items():
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

    main(lmbda)