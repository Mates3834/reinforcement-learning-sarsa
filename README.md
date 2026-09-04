# Reinforcement Learning with SARSA

An implementation and analysis of **SARSA (State–Action–Reward–State–Action)**, an on-policy Temporal Difference (TD) control algorithm, using the **Windy Gridworld** environment.

The project investigates how an agent learns navigation policies through interaction with deterministic and stochastic environments and explores the effects of extended action spaces on learned behavior.

---

## Overview

The project focuses on **on-policy Temporal Difference control**.

The main topics investigated are:

- Reinforcement Learning
- Temporal Difference Learning
- On-Policy Control
- SARSA
- Exploration vs. Exploitation
- ε-Greedy Policies
- Windy Gridworld
- King's Moves
- Stochastic Environments

The objective is to train an agent to reach a target state while minimizing the number of required time steps.

---

## SARSA

SARSA is an on-policy Temporal Difference control algorithm that learns an action-value function:

Q(s,a)

The name SARSA represents the transition sequence:

```text
State → Action → Reward → State → Action
  S       A        R       S'      A'
```

The action-value function is updated according to:

```text
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
```

where:

- `α` — learning rate
- `γ` — discount factor
- `R` — immediate reward
- `Q(S,A)` — current state-action value
- `Q(S',A')` — value of the next state-action pair

Because SARSA is an **on-policy** algorithm, the policy used to interact with the environment is also used during learning.

---

## Exploration vs. Exploitation

An **ε-greedy policy** is used to balance exploration and exploitation.

The agent:

- selects a random action with probability `ε`
- selects the currently preferred action with probability `1-ε`

This allows the agent to explore alternative trajectories while progressively exploiting the learned policy.

---

# Windy Gridworld

The first environment is the classic **Windy Gridworld** control problem.

The agent starts from an initial state and attempts to reach a target state.

However, several columns of the environment contain an upward wind disturbance.

```text
Start
  │
  ▼
Agent ──► Action
            │
            ▼
      Environment
       + Wind
            │
            ▼
        Next State
            │
            ▼
          Reward
            │
            └────► SARSA Update
```

The wind changes the resulting state transition, requiring the agent to learn a path that accounts for environmental disturbances.

---

## Action Space

The initial environment uses four actions:

```text
↑  Up

↓  Down

←  Left

→  Right
```

The objective is to reach the goal using as few time steps as possible.

---

## Reward Structure

The task is formulated as an episodic control problem.

For each time step:

```text
Reward = -1
```

Therefore, policies that reach the target using fewer actions accumulate a higher return.

---

## SARSA Configuration

The baseline implementation uses:

| Parameter | Value |
|---|---:|
| Algorithm | SARSA |
| Policy | ε-Greedy |
| ε | 0.1 |
| α | 0.5 |
| Initial Q-values | 0 |

The agent updates its Q-values continuously while interacting with the environment.

---

# King's Moves

The action space is subsequently extended from four actions to **eight actions**.

In addition to the standard movements, diagonal actions are introduced:

```text
↖   ↑   ↗

←       →

↙   ↓   ↘
```

The diagonal movements are:

- Northeast
- Northwest
- Southeast
- Southwest

The larger action space allows the agent to investigate shorter and more efficient trajectories through the windy environment.

---

## No-Op Action

An additional **No-Op** action is also investigated.

This gives the agent the option to perform no directional movement while still being affected by the wind.

The resulting action space can therefore contain **nine possible actions**.

This experiment investigates whether the agent can exploit environmental dynamics rather than always actively moving.

---

# Stochastic Wind

The environment is further extended by introducing **stochastic wind**.

Instead of having a completely deterministic wind magnitude, the effective wind can vary randomly.

For a nominal wind strength `W`, the actual wind can become:

```text
W - 1
W
W + 1
```

with equal probability.

This creates uncertainty in the environment's state transitions.

---

## Learning under Stochastic Conditions

The SARSA agent continues to update its action-value function after each interaction.

The learning loop can be summarized as:

```text
Initialize Q(s,a)
       │
       ▼
 Select Action
   ε-Greedy
       │
       ▼
 Execute Action
       │
       ▼
Apply Stochastic Wind
       │
       ▼
 Observe S', R
       │
       ▼
 Select A'
       │
       ▼
  SARSA Update
       │
       ▼
   S ← S'
   A ← A'
       │
       └──────── Repeat
```

The objective remains to learn a policy that reaches the target efficiently despite uncertainty in the wind dynamics.

---

# Experiments

The project progressively evaluates SARSA under increasingly complex conditions:

### Experiment 1
**Standard Windy Gridworld**

- Four actions
- Deterministic wind
- ε-greedy SARSA

### Experiment 2
**Windy Gridworld with King's Moves**

- Eight directional actions
- Diagonal movement
- Deterministic wind

### Experiment 3
**King's Moves + No-Op**

- Nine possible actions
- Ability to exploit wind without directional movement

### Experiment 4
**Stochastic Wind**

- Eight-directional action space
- Random wind variation
- Uncertain state transitions

---

# Key Concepts Demonstrated

This project demonstrates:

- Value-based Reinforcement Learning
- Temporal Difference learning
- On-policy control
- State-action value estimation
- ε-greedy exploration
- Policy improvement
- Sequential decision making
- Deterministic vs. stochastic environments
- Extended action-space analysis

---

# Technologies

- Reinforcement Learning
- SARSA
- Temporal Difference Learning
- ε-Greedy Policy
- Python / MATLAB
- Gridworld Simulation

---

# Research Relevance

The project provides a foundation for subsequent work involving:

- Reinforcement Learning for control systems
- Learning-based autonomous navigation
- Decision-making under uncertainty
- Hybrid model-based / learning-based control
- Residual Reinforcement Learning

In particular, the transition from conventional SARSA-based control to **Residual Reinforcement Learning** represents a progression toward combining classical control architectures with learning-based corrective policies.

---

# Repository Structure

```text
reinforcement-learning-sarsa/
│
├── README.md
│
├── src/
│   ├── sarsa/
│   ├── windy_gridworld/
│   ├── kings_moves/
│   └── stochastic_wind/
│
├── results/
│   ├── learning_curves/
│   ├── learned_paths/
│   └── policy_visualization/
│
└── docs/
```

---

# Project Status

Graduate-level Reinforcement Learning project.

The repository presents the implementation and analysis of SARSA-based on-policy control in deterministic and stochastic gridworld environments.
