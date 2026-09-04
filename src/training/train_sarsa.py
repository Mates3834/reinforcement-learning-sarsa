from dataclasses import dataclass
import numpy as np

from src.agents.sarsa_agent import SarsaAgent


@dataclass
class TrainingResult:
    episode_lengths: np.ndarray
    returns: np.ndarray
    q_values: np.ndarray


def train_sarsa(
    env,
    episodes=500,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
    max_steps=10000,
    seed=None,
):
    """
    Train a tabular SARSA agent.

    Returns:
        TrainingResult
    """

    agent = SarsaAgent(
        state_shape=(env.height, env.width),
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
    )

    lengths = np.zeros(episodes, dtype=int)
    returns = np.zeros(episodes, dtype=float)

    for ep in range(episodes):
        state = env.reset()
        action = agent.select_action(state)
        total_reward = 0.0

        for step in range(1, max_steps + 1):
            next_state, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_state)

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                done,
            )

            total_reward += reward
            state, action = next_state, next_action

            if done:
                break

        lengths[ep] = step
        returns[ep] = total_reward

    return TrainingResult(
        episode_lengths=lengths,
        returns=returns,
        q_values=agent.q.copy(),
    )
