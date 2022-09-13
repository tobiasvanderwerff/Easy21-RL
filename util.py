import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


CARD_VALS = list(range(1, 11))

class CardColor(Enum):
    RED = 1
    BLACK = 2
    
    @staticmethod
    def sample() -> "CardColor":
        return CardColor.RED if random.random() < 1 / 3 else CardColor.BLACK


@dataclass(eq=True, frozen=True)
class Card:
    val: int
    color: CardColor

    def __post_init__(self):
        assert self.val in CARD_VALS

    def value(self) -> int:
        return -1 * self.val if self.color is CardColor.RED else self.val
    
    @staticmethod
    def sample() -> "Card":
        val = random.randint(CARD_VALS[0], CARD_VALS[-1])
        color = CardColor.sample()
        return Card(val, color)


class Action(Enum):
    STICK = 1
    HIT = 2


@dataclass(eq=True, frozen=True)
class State:
    dealer_first_card: Card
    player_sum: int
    is_terminal: bool = False


def init_state() -> State:
    # Draw two random black cards at init time.
    dealer_init_val = random.randint(CARD_VALS[0], CARD_VALS[-1])
    player_init_val = random.randint(CARD_VALS[0], CARD_VALS[-1])
    return State(Card(dealer_init_val, CardColor.BLACK), player_init_val)
