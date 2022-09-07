import random

from step import step
from util import State, Card, Action, CardColor, CARD_VALS


n_episodes = 100

total_reward = 0
for _ in range(n_episodes):
    # Draw two random black cards at init time.
    dealer_init_val = random.randint(CARD_VALS[0], CARD_VALS[-1])
    player_init_val = random.randint(CARD_VALS[0], CARD_VALS[-1])
    s = State(Card(dealer_init_val, CardColor.BLACK), player_init_val)

    while not s.is_terminal:
        a = Action.HIT if s.player_sum < 17 else Action.STICK  # simple policy
        s, r = step(s, a)
    total_reward += r

print(f"Total reward after {n_episodes} episodes is {total_reward}.")
