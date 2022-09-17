"""
Question 4: Linear Function Approximation in Easy21
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mc import mc_control
from sarsa import sarsa_lambda


def main(lmbda: float = 0):
    """
    Args:
        lmbda: lambda parameter for SARSA-lambda
    """
    q_mc = mc_control(n_episodes=500000)

    lmbdas = np.arange(0, 1.1, 0.1)
    msas = {lmb: 0 for lmb in lmbdas}
    lambda_to_msa_per_episode = {0.0: None, 1.0: None}
    for lmbda in tqdm(lmbdas, desc="SARSA-lambda for various lambda values"):
        q_sarsa, msas_ = sarsa_lambda(n_episodes=1000, lmbda=lmbda, calculate_msa=True, q_target=q_mc)
        if lmbda in [0.0, 1.0]:
            # Save msa for every episode for lambda=0 and lambda=1.
            lambda_to_msa_per_episode[lmbda] = msas_
        msas[lmbda] = msas_[-1]

    # Plot SARSA-lambda mean-squared error compared to MC value function.
    plt.figure()
    plt.plot(list(msas.keys()), list(msas.values()))
    plt.xlabel("lambda")
    plt.ylabel("mean-squared error")
    plt.title("TD-lambda performance as a function of lambda")
    plt.tight_layout()
    plt.waitforbuttonpress()

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



if __name__ == "__main__":
    lmbda = 1.0

    main(lmbda)