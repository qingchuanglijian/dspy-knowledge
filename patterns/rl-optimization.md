---
pattern_id: P17-RLOptimization
difficulty: expert
source_tutorial: Advanced Optimization / Agent Strategy from dspy.ai
api_modules: []
---

## 1. 核心思想 (Core Concept)

Treat the DSPy program as a policy and use reinforcement learning to optimize its behavior. Unlike supervised optimizers (BootstrapFewShot, MIPROv2) that need labeled examples, RL uses reward signals — which can be delayed, sparse, and non-differentiable. Essential for agent strategies where only the final outcome matters, not the correct intermediate steps. The core philosophy is separation of trajectory generation from gradient estimation: the LLM samples actions, the environment evaluates them, and policy gradients update the decision distribution without ever backpropagating through text.

## 2. 类图与数据流 (Architecture)

```
State (context + task description)
    ↓
dspy.Module (policy) samples strategy / prompt variant
    ↓
dspy.Predict(Signature) → Action (text generation)
    ↓
Environment computes Reward (may be sparse / delayed)
    ↓
REINFORCE: advantage = reward - baseline
    ↓
Policy gradient update on strategy weights
    ↓
Improved policy distribution for next episode
```

Key transformation: state → DSPy module generates action → environment scores outcome → policy parameters updated via sampled trajectory rewards. LLM outputs are non-differentiable; gradients flow through the policy's action distribution, not the text itself.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy
import random

class Act(dspy.Signature):
    """Output a single integer action."""
    state: str = dspy.InputField()
    action: int = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class Policy(dspy.Module):
    def __init__(self):
        self.hints = ["Think big", "Think small", "Think median"]
        self.w = [1.0, 1.0, 1.0]
        self.agent = dspy.Predict(Act)
        self.b = 0.0

    def forward(self, target: int):
        p = [x / sum(self.w) for x in self.w]
        i = random.choices(range(3), weights=p)[0]
        result = self.agent(state=f"{self.hints[i]}. Target: {target}")
        r = max(0.0, 1.0 - abs(result.action - target) / target)
        return result.action, r, i

policy = Policy()
for ep in range(10):
    t = random.randint(20, 80)
    a, r, i = policy(t)
    policy.b = 0.9 * policy.b + 0.1 * r
    policy.w[i] *= 1 + 0.1 * (r - policy.b)
    print(f"Ep {ep}: target={t}, action={a}, reward={r:.2f}, hint={policy.hints[i]}")
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Using RL when labeled data exists

**Wrong:**
```python
# You have 500 labeled examples but choose RL
policy = dspy.Predict(MyTask)
# ... complex REINFORCE loop with handmade reward function ...
```

**Correct:**
```python
# Supervised optimizers are 10-100x more sample-efficient
teleprompter = dspy.MIPROv2(metric=my_metric, num_candidates=10)
optimized = teleprompter.compile(policy, trainset=labeled_data)
```

> **Debugging tip:** If you can write a dataset with input/output pairs, use BootstrapFewShot or MIPROv2. Reserve RL for tasks where the only feedback is a scalar reward on the final outcome (e.g., "did the agent complete the purchase?").

### Anti-pattern 2: Reward function misaligned with true objective

**Wrong:**
```python
def reward(text: str) -> float:
    # Policy learns to game length, not quality
    return len(text) / 1000.0
```

**Correct:**
```python
def reward(text: str, ground_truth: str) -> float:
    # Align with the real business goal
    return f1_score(text, ground_truth)
```

> **Debugging tip:** If the policy finds a shortcut that maximizes reward but fails the real task, your reward function is misaligned. Audit by running the best policy trajectory through a held-out human evaluation.

### Anti-pattern 3: Trying to backprop through text

**Wrong:**
```python
result = policy(state="...")
loss = -reward * result.action  # action is a string / int — not differentiable!
loss.backward()
```

**Correct:**
```python
# Use policy gradients: update the distribution over actions,
# not the action itself. The LLM is a sampler, not a layer.
advantage = reward - baseline
policy.weights[chosen_action] *= (1 + lr * advantage)
```

> **Debugging tip:** If you catch yourself treating LM output as a tensor for `.backward()`, stop. LLM outputs are discrete tokens. Use REINFORCE, PPO, or DPO — all of which update the policy distribution using sampled trajectories, not backprop through text.

## 5. 组合指南 (Composition)

RL optimization is most powerful when the policy is a structured DSPy program and the reward captures a real-world outcome.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| RL + ReAct Agent | Optimize tool-use strategy when only the final task success is rewarded — e.g., the agent gets +1 if it books the correct flight, 0 otherwise |
| RL + Custom Module | Optimize multi-step pipeline weights where each step is a DSPy module and the environment scores the final aggregated output |
| RL + BestOfN | Use RL to learn a selection policy: train a critic module that predicts which of N sampled outputs will score highest |

**Example: RL-optimized ReAct tool selection**
```python
class ToolAgent(dspy.Signature):
    """Choose the next tool to use."""
    observation: str = dspy.InputField()
    tool_name: str = dspy.OutputField()

agent = dspy.Predict(ToolAgent)
# RL loop: reward = 1.0 if task succeeds, 0.0 if max steps reached
# Policy gradient updates which tool the agent picks at each state
```

## 6. 进阶变体 (Advanced Variants)

- **PPO with LLM policy:** Use Proximal Policy Optimization to stabilize training. Maintain a clipped surrogate objective and a separate value network (critic) to reduce gradient variance.
- **REINFORCE with baseline:** The MVP shows a simple moving-average baseline. Production systems use a learned critic (state-value function) or self-critical baseline (greedy decoding score).
- **Actor-Critic for multi-step agents:** The actor is a DSPy ReAct module; the critic is a second DSPy module that estimates expected return from the current state. Train both simultaneously.
- **Reward model training:** When the true reward is expensive to compute (e.g., human judgment), train a small DSPy module as a reward model, then use it for RL training.
- **DPO (Direct Preference Optimization):** Skip explicit RL and optimize the policy directly from pairwise preference data. More stable than PPO for alignment tasks.
- **Multi-agent RL:** Multiple DSPy agents act as policies in a shared environment. Use independent Q-learning or centralized training with decentralized execution for cooperative/competitive tasks.
