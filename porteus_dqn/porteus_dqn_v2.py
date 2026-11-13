# porteus_dqn_v2.py
#
# Revised version of risk_sensitive_dqn.py: a DQN for the stationary Porteus
# tail at risk level lambda_k = lambda * gamma^k (Thoppe et al. 2026; tail of
# Algorithm 1 in the Porteus-tails paper).
#
# Changes vs risk_sensitive_dqn.py (see the comments marked [CHANGE]):
#   1. Loss: no SA-alpha target + MSE. Instead, regress directly on the sample
#      target with the LINEX / Itakura-Saito loss, whose minimizer is exactly
#      the Porteus backup  Q = E_lambda[R + gamma max Q(S')]  (entropic risk).
#      This is your linex_loss(z, ell, c=-1), divided by lambda^2 so it
#      becomes ordinary DQN's 0.5*(Q - y)^2 as lambda -> 0.
#   2. The network outputs Q (not z = -lambda*Q), so value scales are the
#      same for every lambda and lambda = 0 (risk-neutral DQN) is a special case.
#   3. The exponential in the loss is clamped (linear continuation), so it
#      cannot overflow early in training.
#   4. Observation includes visited_bridge (otherwise not Markov), one-hot.
#   5. No gradient-norm clipping (it under-weights the rare -100 falls and
#      inflated Q-values); Double DQN; hard target update every C steps
#      (DQN 2015); epsilon decays over a fraction of the run.
#   6. Exact Porteus solution by value iteration (model is known here), used
#      to check the learned greedy policy.
#
# Usage:
#   python porteus_dqn_v2.py --lam 2            # strongly risk-averse
#   python porteus_dqn_v2.py --lam 0            # risk-neutral DQN
#   python porteus_dqn_v2.py --lam 0.01 --steps 80000

import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from risky_bridge_cliff import RiskyBridgeCliffWorld


@dataclass
class Config:
    gamma: float = 0.99
    risk_lambda: float = 2.0
    stationarity_index: int = 5          # k; tail risk is lambda * gamma^k

    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    train_steps: int = 100_000
    target_update_interval: int = 1000   # hard copy every C steps (DQN 2015)
    grad_clip: float = 0.0               # 0 = off. Clipping biases the fit against
                                         # rare large losses (the -100 falls).
    exp_clamp: float = 5.0               # clamp for exp(.) in the loss
    double_dqn: bool = True

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5  # decay over this fraction of steps

    features: str = "onehot"             # "onehot", "onehot_noflag" or "coords"
    loss: str = "linex"                  # "linex" or "sa" (ablation: your original
                                         # alpha-step target + MSE in z-space)
    sa_alpha: float = 0.01
    hidden_dim: int = 128
    seed: int = 0
    device: str = "cpu"                  # tiny network: CPU is faster


# ------------------------------------------------------------
# Features  [CHANGE 4]
# ------------------------------------------------------------

def make_featurizer(env, kind):
    rows, cols = env.rows, env.cols
    if kind == "onehot":
        dim = rows * cols * 2

        def f(obs):
            v = np.zeros(dim, dtype=np.float32)
            r, c, b = (int(x) for x in obs)
            v[(b * rows + r) * cols + c] = 1.0
            return v
    elif kind == "onehot_noflag":
        # Ablation: hide visited_bridge (the original, non-Markov observation).
        dim = rows * cols

        def f(obs):
            v = np.zeros(dim, dtype=np.float32)
            v[int(obs[0]) * cols + int(obs[1])] = 1.0
            return v
    else:
        dim = 3

        def f(obs):
            r, c, b = obs
            return np.array([r / (rows - 1), c / (cols - 1), b],
                            dtype=np.float32)
    return f, dim


# ------------------------------------------------------------
# Loss  [CHANGE 1, 3]
# ------------------------------------------------------------

def porteus_loss(q_pred, y, lam, clamp):
    """Loss whose minimizer over q is the entropic risk E_lam[y].

    With z = -lam*q and ell = -lam*y (your notation), d = ell - z and the loss
    is exp(d) - d - 1, i.e. linex_loss(z, ell, c=-1). Its gradient in z is
    1 - exp(ell - z), the same direction as the SA step x <- x + a(F - x).
    Divided by lam^2 it tends to 0.5*(q - y)^2 as lam -> 0.
    For d > clamp the exponential is continued linearly (no overflow).
    """
    if abs(lam) < 1e-8:
        return 0.5 * ((q_pred - y) ** 2).mean()
    d = lam * (q_pred - y)
    d_c = torch.clamp(d, max=clamp)
    slope = np.exp(clamp) - 1.0
    loss = torch.exp(d_c) - d_c - 1.0 + slope * (d - d_c)
    return loss.mean() / lam ** 2


# ------------------------------------------------------------
# Network and replay
# ------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, in_dim, num_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)  # Q-values; z = -lambda_k * Q if you need it


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, *transition):
        self.buf.append(transition)

    def sample(self, n, device):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        return (t(s, torch.float32), t(a, torch.long), t(r, torch.float32),
                t(s2, torch.float32), t(d, torch.float32))

    def __len__(self):
        return len(self.buf)


# ------------------------------------------------------------
# Agent
# ------------------------------------------------------------

class PorteusTailDQN:
    def __init__(self, in_dim, num_actions, cfg):
        self.cfg = cfg
        self.num_actions = num_actions
        self.device = torch.device(cfg.device)
        self.lambda_k = cfg.risk_lambda * cfg.gamma ** cfg.stationarity_index
        self.online = QNetwork(in_dim, num_actions, cfg.hidden_dim).to(self.device)
        self.target = QNetwork(in_dim, num_actions, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.updates = 0

    def epsilon(self, step):
        cfg = self.cfg
        frac = min(1.0, step / (cfg.epsilon_decay_fraction * cfg.train_steps))
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    @torch.no_grad()
    def q_values(self, feat):
        x = torch.as_tensor(feat, device=self.device).unsqueeze(0)
        return self.online(x)[0].cpu().numpy()

    def act(self, feat, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        q = self.q_values(feat)
        return int(random.choice(np.flatnonzero(q == q.max())))

    def train_step(self):
        cfg = self.cfg
        s, a, r, s2, done = self.replay.sample(cfg.batch_size, self.device)
        q = self.online(s).gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if cfg.double_dqn:
                a2 = self.online(s2).argmax(1, keepdim=True)
                q_next = self.target(s2).gather(1, a2).squeeze(1)
            else:
                q_next = self.target(s2).max(1).values
            # Sample target. Terminal: future value 0 (x = 1). Truncation by
            # time limit is not stored as done, so it still bootstraps.
            y = r + cfg.gamma * (1.0 - done) * q_next

        if cfg.loss == "sa":
            # Ablation: the original risk_sensitive_dqn.py update, in z = -lam*Q:
            # z_target = log((1-alpha) e^{z_old} + alpha e^{ell}), then MSE.
            lam = self.lambda_k
            z = -lam * q
            ell = -lam * y
            with torch.no_grad():
                z_target = torch.logaddexp(
                    np.log1p(-cfg.sa_alpha) + z.detach(),
                    np.log(cfg.sa_alpha) + ell)
            loss = ((z - z_target) ** 2).mean()
        else:
            loss = porteus_loss(q, y, self.lambda_k, cfg.exp_clamp)
        self.opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:  # [CHANGE 5] off by default, see Config
            nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.opt.step()

        self.updates += 1
        if self.updates % cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()


# ------------------------------------------------------------
# Exact Porteus solution (value iteration on the known model)  [CHANGE 6]
# ------------------------------------------------------------

def exact_porteus_q(env, lam, gamma, tol=1e-10, max_iter=100_000):
    """Fixed point of Q(s,a) = E_lam[R + gamma * max_a' Q(S',a')] on
    augmented states (row, col, visited_bridge). Terminal value is 0."""
    states = [(r, c, b) for b in (0, 1) for r in range(env.rows)
              for c in range(env.cols)
              if (r, c) not in env.cliff and (r, c) != env.goal]
    idx = {s: i for i, s in enumerate(states)}
    p = env.slip_probability

    # outcomes[i][a] = list of (prob, reward, next_index or None if terminal)
    outcomes = []
    for (r, c, b) in states:
        per_action = []
        for a in range(4):
            dr, dc = env.moves[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.rows and 0 <= nc < env.cols):
                nr, nc = r, c
            if (nr, nc) == env.bridge:
                per_action.append([(p, -100.0, None),
                                   (1 - p, -1.0, idx[(nr, nc, 1)])])
            elif (nr, nc) in env.cliff:
                per_action.append([(1.0, -100.0, None)])
            elif (nr, nc) == env.goal:
                per_action.append([(1.0, 20.0 if b else 6.0, None)])
            else:
                per_action.append([(1.0, -1.0, idx[(nr, nc, b)])])
        outcomes.append(per_action)

    Q = np.zeros((len(states), 4))
    for _ in range(max_iter):
        V = Q.max(1)
        Q_new = np.empty_like(Q)
        for i in range(len(states)):
            for a in range(4):
                probs = np.array([o[0] for o in outcomes[i][a]])
                vals = np.array([o[1] + (0.0 if o[2] is None else gamma * V[o[2]])
                                 for o in outcomes[i][a]])
                if abs(lam) < 1e-12:
                    Q_new[i, a] = probs @ vals
                else:
                    # -(1/lam) log sum p exp(-lam v), computed stably
                    m = (-lam * vals).max()
                    Q_new[i, a] = -(m + np.log(probs @ np.exp(-lam * vals - m))) / lam
        if np.abs(Q_new - Q).max() < tol:
            Q = Q_new
            break
        Q = Q_new
    return states, Q


# ------------------------------------------------------------
# Train / evaluate
# ------------------------------------------------------------

def train(cfg):
    # The network is tiny, so splitting each matrix multiply across many CPU
    # threads costs more than it saves (and keeps ~14 cores busy-waiting).
    torch.set_num_threads(1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = RiskyBridgeCliffWorld(slip_probability=0.1,
                                include_bridge_flag=True, verbose=False)
    feat, in_dim = make_featurizer(env, cfg.features)
    agent = PorteusTailDQN(in_dim, env.action_space.n, cfg)

    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, recent, losses = 0.0, deque(maxlen=50), deque(maxlen=500)
    for step in range(1, cfg.train_steps + 1):
        eps = agent.epsilon(step)
        a = agent.act(feat(obs), eps)
        obs2, r, term, trunc, _ = env.step(a)
        agent.replay.add(feat(obs), a, r, feat(obs2), float(term))
        obs = obs2
        ep_ret += r
        if step >= cfg.warmup_steps:
            losses.append(agent.train_step())
        if term or trunc:
            recent.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        if step % 5000 == 0:
            print(f"step {step:6d} | eps {eps:.2f} | "
                  f"mean return (last 50 eps) {np.mean(recent) if recent else float('nan'):7.2f} | "
                  f"loss {np.mean(losses) if losses else float('nan'):.4f}")
    return env, feat, agent


def evaluate(env, feat, agent, cfg, episodes=1000):
    events = {"goal (safe)": 0, "goal (risky)": 0, "bridge failure": 0,
              "cliff": 0, "timeout": 0}
    returns, disc_returns = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=100_000 + ep)
        G, Gd, t = 0.0, 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(agent.act(feat(obs)))
            G += r
            Gd += cfg.gamma ** t * r
            t += 1
            if term or trunc:
                break
        ev = info.get("event")
        if ev == "goal":
            events[f"goal ({info['route']})"] += 1
        elif ev in ("bridge_failure", "cliff"):
            events[ev.replace("_", " ")] += 1
        else:
            events["timeout"] += 1
        returns.append(G)
        disc_returns.append(Gd)

    Gd = np.array(disc_returns)
    lam = cfg.risk_lambda
    if abs(lam) > 1e-12:
        m = (-lam * Gd).max()
        ent = -(m + np.log(np.mean(np.exp(-lam * Gd - m)))) / lam
    else:
        ent = Gd.mean()
    print(f"\nGreedy evaluation over {episodes} episodes:")
    for k, v in events.items():
        print(f"  {k:15s} {v}")
    print(f"  mean return            {np.mean(returns):.2f}")
    print(f"  mean discounted return {Gd.mean():.2f}")
    print(f"  entropic risk (lambda={lam}) of discounted return {ent:.2f}"
          "  (Monte Carlo; slightly optimistic for finite samples)")


def compare_with_exact(env, feat, agent, cfg):
    states, Q_exact = exact_porteus_q(env, agent.lambda_k, cfg.gamma)
    agree, worst = 0, 0.0
    for i, s in enumerate(states):
        a = int(np.argmax(agent.q_values(feat(np.array(s)))))
        gap = Q_exact[i].max() - Q_exact[i, a]   # loss in exact Porteus value
        agree += gap < 1e-6
        worst = max(worst, gap)
    i0 = states.index((*env.start, 0))
    print(f"\nExact Porteus solution (lambda_k = {agent.lambda_k:.4f}):")
    print(f"  learned greedy action is exactly optimal in {agree}/{len(states)} states"
          f" (worst value gap {worst:.3f})")
    print(f"  start state: exact max Q = {Q_exact[i0].max():.2f}, "
          f"learned max Q = {agent.q_values(feat(np.array((*env.start, 0)))).max():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--features", choices=["onehot", "onehot_noflag", "coords"],
                   default="onehot")
    p.add_argument("--loss", choices=["linex", "sa"], default="linex")
    p.add_argument("--sa-alpha", type=float, default=0.01)
    p.add_argument("--no-ddqn", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-update", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default="porteus_dqn_v2.pt")
    args = p.parse_args()

    cfg = Config(risk_lambda=args.lam, stationarity_index=args.k,
                 train_steps=args.steps, features=args.features,
                 double_dqn=not args.no_ddqn, seed=args.seed, device=args.device,
                 learning_rate=args.lr, target_update_interval=args.target_update,
                 grad_clip=args.grad_clip, loss=args.loss,
                 sa_alpha=args.sa_alpha)
    env, feat, agent = train(cfg)
    evaluate(env, feat, agent, cfg)
    compare_with_exact(env, feat, agent, cfg)
    torch.save({"online": agent.online.state_dict(), "config": cfg.__dict__},
               args.save)
    print(f"\nSaved model to {args.save}")


if __name__ == "__main__":
    main()
