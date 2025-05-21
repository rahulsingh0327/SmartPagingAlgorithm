import numpy as np
import pandas as pd
import random

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ----- Set Random Seed -----
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ----- Data Preprocessing -----
# Load and fix CSV formatting
with open('windows.csv','r') as f:
    lines = f.readlines()
header = lines[0].strip().split(',')
raw_lines = lines[1:]
fixed_lines, buffer = [], ""
for line in raw_lines:
    if buffer:
        buffer += line.strip()
        if buffer.count(',') == len(header)-1:
            fixed_lines.append(buffer)
            buffer = ""
        else:
            continue
    else:
        if line.count(',') == len(header)-1:
            fixed_lines.append(line.strip())
        else:
            buffer = line.strip()
data = pd.DataFrame([l.split(',') for l in fixed_lines], columns=header)
for col in ['Timestamp','RAM_Usage_MB','Swap_Usage_MB','CPU_Usage','Page_Faults_Delta']:
    dtype = float if col!='Page_Faults_Delta' else int
    data[col] = data[col].astype(dtype)

# Label actions based on RAM change
eps = 1.0
actions = []
for i in range(len(data)-1):
    diff = data.loc[i+1,'RAM_Usage_MB'] - data.loc[i,'RAM_Usage_MB']
    actions.append(2 if diff > eps else 1 if diff < -eps else 0)
actions.append(0)
data['action'] = actions

# Create transitions (s,a,s',r)
states = data[['RAM_Usage_MB','Swap_Usage_MB','CPU_Usage']].values[:-1]
next_states = data[['RAM_Usage_MB','Swap_Usage_MB','CPU_Usage']].values[1:]
actions = np.array(actions[:-1])
rewards = -data['Page_Faults_Delta'].to_numpy(dtype=np.float32)[1:]  # Fix conversion

# Normalize states
scaler = StandardScaler()
states = scaler.fit_transform(states)
next_states = scaler.transform(next_states)

# ----- Environment Proxy -----
class MemoryEnv:
    def __init__(self, states, actions, next_states, rewards):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.action_indices = {a: np.where(actions == a)[0] for a in np.unique(actions)}

    def reset(self):
        return self.states[0]

    def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        idxs = self.action_indices.get(action, [])
        if len(idxs) == 0:
            return state, 0.0
        dist = np.sum((self.states[idxs] - state)**2, axis=1)
        best = idxs[np.argmin(dist)]
        return self.next_states[best], self.rewards[best]

env = MemoryEnv(states, actions, next_states, rewards)

# ----- Q-Network and Training -----
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

state_tensor = torch.tensor(states, dtype=torch.float32)
next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
action_tensor = torch.tensor(actions, dtype=torch.long)
reward_tensor = torch.tensor(rewards, dtype=torch.float32)

print("Reward stats:", reward_tensor.min().item(), reward_tensor.max().item(), reward_tensor.mean().item())
print("Non-zero rewards count:", (reward_tensor != 0).sum().item())

dataset = TensorDataset(state_tensor, action_tensor, reward_tensor, next_state_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

state_dim, action_dim = states.shape[1], 3
Q = QNetwork(state_dim, action_dim)
Q_target = QNetwork(state_dim, action_dim)
Q_target.load_state_dict(Q.state_dict())

optimizer = torch.optim.Adam(Q.parameters(), lr=1e-3)
gamma = 0.99

for epoch in range(50):
    for s_batch, a_batch, r_batch, s2_batch in loader:
        q_vals = Q(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next = Q_target(s2_batch).max(dim=1).values
            target = r_batch + gamma * max_next
        loss = nn.MSELoss()(q_vals, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        Q_target.load_state_dict(Q.state_dict())

# ----- Evaluation -----
def simulate_policy(policy_fn, env, start_state, steps=1000):
    state = start_state.copy()
    total_pf = 0.0
    for _ in range(steps):
        action = policy_fn(state)
        next_state, rew = env.step(state, action)
        total_pf += -rew
        state = next_state
    return total_pf

baseline_policy = lambda s: 0
def rl_policy(s):
    q_vals = Q(torch.tensor(s, dtype=torch.float32))
    return int(torch.argmax(q_vals).item())

start = env.reset()
baseline_faults = simulate_policy(baseline_policy, env, start, steps=1000)
rl_faults = simulate_policy(rl_policy, env, start, steps=1000)

print(f"Baseline faults: {baseline_faults:.1f}, RL faults: {rl_faults:.1f}")
if baseline_faults > 0:
    reduction = 100 * (baseline_faults - rl_faults) / baseline_faults
    print(f"Reduction: {reduction:.1f}%")

# ----- Inspect Q-values -----
print("\nSample Q-values:")
for i in range(5):
    q = Q(torch.tensor(states[i], dtype=torch.float32)).detach().numpy()
    print(f"State {i}: {states[i]} → Q-values: {q}")