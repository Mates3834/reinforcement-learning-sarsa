import numpy as np


class WindyGridworld:
    """
    Generic Windy Gridworld environment.

    State:
        (row, col)

    Reward:
        -1 per transition until the goal is reached.

    Variants:
        - 4 actions
        - King's Moves (8 actions)
        - optional No-Op
        - optional stochastic wind
    """

    ACTIONS_4 = [
        (-1, 0),   # up
        (1, 0),    # down
        (0, -1),   # left
        (0, 1),    # right
    ]

    ACTIONS_8 = ACTIONS_4 + [
        (-1, -1),  # up-left
        (-1, 1),   # up-right
        (1, -1),   # down-left
        (1, 1),    # down-right
    ]

    def __init__(
        self,
        height=7,
        width=10,
        start=(3, 0),
        goal=(3, 7),
        wind=None,
        kings_moves=False,
        no_op=False,
        stochastic_wind=False,
        seed=None,
    ):
        self.height = height
        self.width = width
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.stochastic_wind = stochastic_wind
        self.rng = np.random.default_rng(seed)

        if wind is None:
            wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.wind = np.asarray(wind, dtype=int)

        if len(self.wind) != self.width:
            raise ValueError("Wind vector length must equal grid width.")

        self.actions = list(self.ACTIONS_8 if kings_moves else self.ACTIONS_4)
        if no_op:
            self.actions.append((0, 0))

        self.state = self.start

    @property
    def n_actions(self):
        return len(self.actions)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action_index):
        row, col = self.state
        dr, dc = self.actions[action_index]

        wind_strength = int(self.wind[col])
        if self.stochastic_wind and wind_strength > 0:
            wind_strength += int(self.rng.choice([-1, 0, 1]))

        next_row = row + dr - wind_strength
        next_col = col + dc

        next_row = int(np.clip(next_row, 0, self.height - 1))
        next_col = int(np.clip(next_col, 0, self.width - 1))

        self.state = (next_row, next_col)
        done = self.state == self.goal
        reward = -1.0

        return self.state, reward, done, {}
