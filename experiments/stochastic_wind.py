from src.environment.windy_gridworld import WindyGridworld
from src.training.train_sarsa import train_sarsa


def run():
    env = WindyGridworld(
        kings_moves=True,
        no_op=False,
        stochastic_wind=True,
        seed=4,
    )
    result = train_sarsa(
        env,
        episodes=700,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1,
        seed=4,
    )
    print("Stochastic wind")
    print("Mean episode length, last 50:",
          result.episode_lengths[-50:].mean())


if __name__ == "__main__":
    run()
