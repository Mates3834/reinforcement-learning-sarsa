from src.environment.windy_gridworld import WindyGridworld
from src.training.train_sarsa import train_sarsa


def run():
    env = WindyGridworld(kings_moves=True, no_op=False, stochastic_wind=False)
    result = train_sarsa(
        env,
        episodes=500,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1,
        seed=2,
    )
    print("King's Moves")
    print("Mean episode length, last 50:",
          result.episode_lengths[-50:].mean())


if __name__ == "__main__":
    run()
