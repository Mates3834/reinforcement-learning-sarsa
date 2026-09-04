import matplotlib.pyplot as plt

from src.environment.windy_gridworld import WindyGridworld
from src.training.train_sarsa import train_sarsa


def moving_average(x, window=20):
    if len(x) < window:
        return x
    import numpy as np
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


env = WindyGridworld(
    kings_moves=True,
    no_op=False,
    stochastic_wind=False,
)

result = train_sarsa(
    env,
    episodes=500,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
    seed=0,
)

smoothed = moving_average(result.episode_lengths, 20)

plt.figure()
plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("SARSA Learning Curve")
plt.grid(True)
plt.show()
