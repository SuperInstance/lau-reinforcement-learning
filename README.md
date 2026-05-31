# lau-reinforcement-learning

A reinforcement learning library implementing MDP formalism, dynamic programming, temporal difference learning, Q-learning, policy gradient methods, multi-armed bandits, and grid world environments.

## Features

- **MDP Formalism**: States, actions, transitions, rewards, and discounting via a generic `MDP` trait
- **Value Functions**: State-value (V) and action-value (Q) functions with incremental updates
- **Dynamic Programming**: Policy evaluation, policy iteration, and value iteration
- **Temporal Difference Learning**: TD(0) and TD(λ) with eligibility traces
- **Q-Learning**: Tabular Q-learning with ε-greedy exploration
- **Policy Gradients**: REINFORCE with softmax policies
- **Multi-Armed Bandits**: ε-greedy, UCB1, and Thompson Sampling agents
- **Grid World Environments**: Configurable grid worlds with walls, goals, penalties, and stochastic transitions

## Usage

```rust
use lau_reinforcement_learning::*;

// Create a grid world
let gw = gridworld::GridWorld::standard_4x4();

// Solve with value iteration
let (policy, v) = dp::value_iteration(&gw, 1e-10, 1000);

// Get the optimal action for state 0
if let Some(action) = policy.get(0) {
    println!("Optimal action at (0,0): {:?}", action);
}
```

## Running Tests

```bash
cargo test
```

## License

MIT
