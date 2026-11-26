# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.


## Problem Statement

Implement a model-free Q-Learning algorithm using Gymnasium's FrozenLake-v1 environment. Train the agent using an epsilon-greedy strategy and update the Q-table using the Q-Learning update rule. Gradually reduce exploration and learn a policy to reach the goal while avoiding holes. Display the final Q-table, state-value function, learned policy, learning curve, and average reward over the last 1000 episodes.

## Software Requirements

Programming Language
Python 3
Libraries
The following Python libraries are required:

```
gymnasium
numpy
matplotlib
```

Install the required packages using:

```
pip install gymnasium numpy matplotlib
```


## Environment Description
FrozenLake-v1 is an 8 × 8 grid-world environment with 64 states.

- S – Start
- F – Frozen (Safe)
- H – Hole
- G – Goal

It has four actions: Left (0), Down (1), Right (2), Up (3).

The environment uses `is_slippery=True`, making movements stochastic (unpredictable).


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
1. Create the FrozenLake-v1 environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros.
4. Set learning rate (α), discount factor (γ), and exploration rate (ε).
5. Repeat for the specified number of episodes:
   - Reset the environment.
   - Select an action using epsilon-greedy strategy.
   - Execute the action and observe reward and next state.
   - Update the Q-value using the Q-Learning formula.
   - Continue until the episode terminates.
   - Store the total reward and reduce epsilon.
6. Calculate the state-value function and greedy policy.
7. Display the Q-table, state-value function, and learned policy.
8. Calculate the average reward over the last 1000 episodes.
9. Plot the learning curve.
10. Close the environment.


## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

```


## Output

### Final Q-table:
<img width="452" height="609" alt="Screenshot 2026-11-26 134957" src="https://github.com/user-attachments/assets/a7ae7a99-ea7d-4445-9235-18210c65a7b1" />


### Estimated State-Value Function:
<img width="787" height="226" alt="Screenshot 2026-11-26 135008" src="https://github.com/user-attachments/assets/5ff3632d-652a-4832-a6ac-fe3138ee9766" />

### Learned Policy:

<img width="866" height="140" alt="Screenshot 2026-11-26 135015" src="https://github.com/user-attachments/assets/cfbaee63-cb0f-44ad-880a-0d0ab543b95d" />

<img width="993" height="581" alt="Screenshot 2026-11-26 135036" src="https://github.com/user-attachments/assets/61accaf0-1269-4cd7-8a88-d6d957ef827b" />


## Result

Thus, the Q-Learning algorithm was successfully implemented using the FrozenLake-v1 environment with an 8 × 8 grid. The agent learned a policy through repeated interactions, and the final Q-table, state-value function, learned policy, learning curve, and average reward were obtained.


## Inference

The experiment demonstrates that Q-Learning is a model-free control algorithm that learns a suitable policy through trial and error without requiring a predefined environment model. Through exploration and exploitation, the agent improves its Q-values and learns to reach the goal while avoiding holes in the stochastic FrozenLake environment.
