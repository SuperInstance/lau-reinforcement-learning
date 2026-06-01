# lau-reinforcement-learning

A pure-Rust reinforcement learning library implementing MDP formalism, dynamic programming, temporal-difference learning, Q-learning, REINFORCE policy gradients, multi-armed bandits, eligibility traces, and grid-world environments.

**57 tests** · `nalgebra` + `serde` + `rand` · MIT license

---

## What This Does

`lau-reinforcement-learning` provides the foundational algorithms of reinforcement learning, from exact dynamic programming (policy evaluation, policy iteration, value iteration) through sample-based methods (TD, Q-learning, REINFORCE) to bandit algorithms (ε-greedy, UCB1, Thompson Sampling).

Everything is built on a generic `MDP` trait, so you can define your own environments and immediately use every algorithm in the crate. A built-in `GridWorld` environment gives you a ready-made testbed.

---

## Key Idea

Reinforcement learning is about agents learning optimal behaviour through interaction. This crate encodes the standard algorithms from Sutton & Barto's *Reinforcement Learning: An Introduction* as composable Rust functions:

| Module | What | Algorithm |
|---|---|---|
| `mdp` | MDP formalism | `State`, `Action`, `Transition`, `TabularMDP`, policies |
| `value` | Value functions | V(s), Q(s,a) — tabular storage |
| `dp` | Dynamic programming | Policy evaluation, policy iteration, value iteration |
| `td` | Temporal difference | TD(0), TD(λ) with eligibility traces |
| `q_learning` | Q-learning | Tabular Q-learning, ε-greedy, greedy policy extraction |
| `policy_gradient` | Policy gradients | REINFORCE with softmax policy |
| `bandit` | Multi-armed bandits | ε-greedy, UCB1, Thompson Sampling |
| `eligibility` | Eligibility traces | Accumulating & replacing traces |
| `gridworld` | Grid-world environments | 4×4, cliff walking, custom grids |

---

## Install

```toml
[dependencies]
lau-reinforcement-learning = "0.1"
```

Or:

```sh
cargo add lau-reinforcement-learning
```

---

## Quick Start

### Value Iteration on a Grid World

```rust
use lau_reinforcement_learning::gridworld::GridWorld;
use lau_reinforcement_learning::dp::value_iteration;
use lau_reinforcement_learning::mdp::MDP;

let mdp = GridWorld::standard_4x4();  // 4×4 grid, terminal at (3,3)
let (policy, v) = value_iteration(&mdp, 1e-6, 1000);

for state in mdp.states() {
    if let Some(action) = policy.get(state) {
        println!("State {}: go {:?}, V = {:.3}", state, action, v.get(&state));
    }
}
```

### Q-Learning with ε-Greedy Exploration

```rust
use lau_reinforcement_learning::gridworld::GridWorld;
use lau_reinforcement_learning::q_learning::*;
use lau_reinforcement_learning::value::ActionValueFunction;
use lau_reinforcement_learning::mdp::MDP;
use rand::thread_rng;

let mdp = GridWorld::cliff_walking();
let mut q = ActionValueFunction::new(0.0);
let mut rng = thread_rng();

for episode in 0..500 {
    let start = mdp.states()[0];
    let eps = 0.1;
    let policy_fn = |s, rng: &mut _| {
        let actions = mdp.actions(s);
        epsilon_greedy(&q, &s, &actions, eps, rng)
    };
    let (total_reward, steps) = q_learning_episode(
        &mdp, &mut q, &policy_fn, start, 0.1, 200, &mut rng,
    );
    if episode % 100 == 0 {
        println!("Episode {}: reward={:.1}, steps={}", episode, total_reward, steps);
    }
}

let greedy = extract_greedy_policy(&mdp, &q);
```

### Multi-Armed Bandit: Thompson Sampling vs ε-Greedy

```rust
use lau_reinforcement_learning::bandit::*;
use rand::thread_rng;

let bandit = GaussianBandit::new(vec![1.0, 2.0, 3.0, 2.5], 1.0);
let mut rng = thread_rng();

let result_ts = run_thompson_sampling(&bandit, 1000, &mut rng);
let result_eg = run_epsilon_greedy(&bandit, 0.1, 1000, &mut rng);

println!("Thompson: total reward = {:.1}, optimal pull rate = {:.1}%",
    result_ts.total_reward(), result_ts.optimal_pull_rate() * 100.0);
println!("ε-greedy: total reward = {:.1}, optimal pull rate = {:.1}%",
    result_eg.total_reward(), result_eg.optimal_pull_rate() * 100.0);
```

### REINFORCE Policy Gradient

```rust
use lau_reinforcement_learning::policy_gradient::*;
use lau_reinforcement_learning::mdp::MDP;
use lau_reinforcement_learning::gridworld::GridWorld;
use rand::thread_rng;

let mdp = GridWorld::standard_4x4();
let mut policy = SoftmaxPolicy::new(0.01);  // learning rate α
let mut rng = thread_rng();
let actions: Vec<_> = mdp.all_actions();

for _ in 0..1000 {
    let traj = collect_trajectory(
        &mdp,
        &|s, rng| policy.sample(&s, &actions, rng),
        0,
        100,
        &mut rng,
    );
    reinforce_update(&mut policy, &traj, 0.99, &actions);
}
```

---

## API Reference

### `mdp` — Markov Decision Processes

| Type | Description |
|---|---|
| `trait State` | Marker for state types (must be `Copy + Eq + Hash`) |
| `trait Action` | Marker for action types |
| `struct Transition<S, A>` | (next_state, reward, probability) |
| `trait MDP` | Core trait: `states()`, `actions(s)`, `transitions(s,a)`, `discount_factor()`, `is_terminal(s)` |
| `TabularMDP` | Concrete MDP with indexed states/actions |
| `StochasticPolicy<S, A>` | Maps states → probability distribution over actions |
| `DeterministicPolicy<S, A>` | Maps states → single action |

### `value` — Value Functions

| Type | Key Methods |
|---|---|
| `StateValueFunction<S>` | `new(default)`, `get(s)`, `set(s, v)`, `update(s, delta)` |
| `ActionValueFunction<S, A>` | `new(default)`, `get(s, a)`, `set(s, a, v)`, `greedy_action(s, actions)`, `max_q(s, actions)` |

### `dp` — Dynamic Programming

| Function | Signature | Description |
|---|---|---|
| `policy_evaluation_stochastic` | `(mdp, policy, θ, max_iter) → V` | Iterative policy eval for stochastic policies |
| `policy_evaluation_deterministic` | `(mdp, policy, θ, max_iter) → V` | Same, for deterministic policies |
| `policy_iteration` | `(mdp, θ, max_iter) → (π, V)` | Alternate eval + improve until stable |
| `value_iteration` | `(mdp, θ, max_iter) → (π, V)` | Direct optimal V via Bellman optimality |
| `v_to_q` | `(mdp, V) → Q` | Convert state-value to action-value function |

### `td` — Temporal Difference Learning

| Function | Description |
|---|---|
| `td0_update` | Single TD(0) update: V(s) ← V(s) + α[r + γV(s') − V(s)] |
| `td_lambda_update` | TD(λ) with accumulating or replacing traces |
| `td0_episode` | Run a full TD(0) episode on an MDP |

### `q_learning` — Q-Learning

| Function | Description |
|---|---|
| `q_learning_update` | Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) − Q(s,a)] |
| `epsilon_greedy` | ε-greedy action selection from Q |
| `q_learning_episode` | Run a full episode, return (total_reward, steps) |
| `extract_greedy_policy` | Derive π* from Q by greedy selection |

### `policy_gradient` — REINFORCE

| Type / Function | Description |
|---|---|
| `SoftmaxPolicy<S, A>` | θ-parameterised softmax: π(a\|s) = e^{θ(s,a)} / Σ e^{θ(s,a')} |
| `Trajectory<S, A>` | Stores (s₀,a₀,r₁,s₁,…), computes discounted returns |
| `collect_trajectory` | Roll out one episode under a given policy |
| `reinforce_update` | Apply REINFORCE: θ(s,a) ← θ(s,a) + α·G·∇ ln π(a\|s) |

### `bandit` — Multi-Armed Bandits

| Type / Function | Description |
|---|---|
| `trait Bandit` | Interface: `pull(arm, rng)`, `num_arms()`, `true_means()` |
| `GaussianBandit` | Each arm ~ N(μᵢ, σ) |
| `EpsilonGreedyAgent` | ε-greedy with incremental Q-estimates |
| `UCB1Agent` | UCB1: select arm maximising Q_a + c√(ln t / n_a) |
| `ThompsonSamplingAgent` | Bayesian: sample from Gaussian-Gamma posterior |
| `run_epsilon_greedy` / `run_ucb1` / `run_thompson_sampling` | Full experiment runners → `BanditResult` |

`BanditResult` contains per-step rewards, optimal pull indicators, cumulative reward, and cumulative regret.

### `eligibility` — Eligibility Traces

```rust
let mut traces = EligibilityTraces::new();
traces.update(&state, 1.0);       // accumulating
traces.set(&state, 1.0);          // replacing
traces.decay(gamma * lambda);     // decay all by γλ
```

### `gridworld` — Grid World Environments

| Method | Description |
|---|---|
| `GridWorld::new(w, h)` | Empty grid |
| `GridWorld::standard_4x4()` | Classic 4×4, goal at (3,3), cost −1 per step |
| `GridWorld::cliff_walking()` | 12×4 cliff walking (Sutton & Barto Ch. 6) |
| `.set_cell(x, y, cell)` | Set Empty / Wall / Goal(r) / Penalty(r) |
| `.slip_probability` | Stochastic transitions (slip to perpendicular) |

Implements the `MDP` trait, so it works with every algorithm.

---

## How It Works

### The MDP Trait

All algorithms are generic over `MDP`:

```rust
pub trait MDP: Send + Sync {
    type S: State;
    type A: Action;
    fn states(&self) -> Vec<Self::S>;
    fn actions(&self, state: Self::S) -> Vec<Self::A>;
    fn transitions(&self, state: Self::S, action: Self::A) -> Vec<Transition<Self::S, Self::A>>;
    fn discount_factor(&self) -> f64;
    fn is_terminal(&self, state: Self::S) -> bool;
    fn all_actions(&self) -> Vec<Self::A>;
}
```

Implement this trait for your environment and every algorithm (DP, TD, Q-learning, REINFORCE) works out of the box.

### Policy Representation

Two flavours:
- **Stochastic**: `HashMap<S, HashMap<A, f64>>` — action probabilities per state.
- **Deterministic**: `HashMap<S, A>` — single action per state.

Both are serialisable with `serde`.

### Bandit Experiments

Each bandit runner returns a `BanditResult` with:
- `rewards`: per-step rewards
- `optimal_pulls`: boolean flags for whether the best arm was chosen
- `cumulative_reward` / `cumulative_regret`: running totals

Thompson Sampling uses a **Normal-Gamma conjugate posterior**: for each arm, it maintains (μ, κ, α, β) hyperparameters and samples precision from Gamma, then mean from Normal.

---

## The Math

### Bellman Equations

**State-value (policy evaluation):**
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [r(s,a,s') + γV^π(s')]

**Action-value (Q-learning):**
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]

### TD(λ)

TD error: **δ_t = r_{t+1} + γV(s_{t+1}) − V(s_t)**

For all states: **V(s) ← V(s) + α · δ_t · e_t(s)**

Traces decay as **e_t(s) = γλ · e_{t-1}(s) + 𝟙{s = s_t}** (accumulating) or **e_t(s) = max(γλ · e_{t-1}(s), 𝟙{s = s_t})** (replacing).

### REINFORCE Gradient

For a softmax policy π_θ(a|s), the policy gradient with Monte Carlo return G_t:

**∇θ J ≈ G_t · ∇θ ln π_θ(a_t|s_t)**

Implemented as:
- If a = a_t: θ(s,a) ← θ(s,a) + α · G_t · (1 − π(a|s))
- Otherwise: θ(s,a) ← θ(s,a) − α · G_t · π(a|s)

### UCB1

Select arm maximising **Q_a + c · √(ln t / n_a)**, where c controls the exploration-exploitation balance. Guarantees O(√(nK log n)) cumulative regret for K arms.

---

## License

MIT
