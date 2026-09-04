import numpy as np


class SarsaAgent:
    """
    Tabular SARSA agent with epsilon-greedy policy.

    Update:
        Q(S,A) <- Q(S,A) + alpha *
                  [R + gamma*Q(S',A') - Q(S,A)]
    """

    def __init__(
        self,
        state_shape,
        n_actions,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1,
        seed=None,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

        self.q = np.zeros((*state_shape, n_actions), dtype=float)

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[-1]))

        values = self.q[state]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, next_action, done):
        q_sa = self.q[state + (action,)]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q[next_state + (next_action,)]

        td_error = target - q_sa
        self.q[state + (action,)] += self.alpha * td_error

        return td_error
