# simulate.py
#
# Visualize a trained agent on RiskyBridgeCliffWorld as an animated GIF.
# Loads an already-trained agent; does not train anything.
#
#   python simulate.py --agent tabular     # tabular_z_values.csv  (tabular_porteus_tail.py)
#   python simulate.py --agent dqn         # porteus_tail_dqn.pt   (risk_sensitive_dqn.py)
#   python simulate.py --agent v2          # porteus_dqn_v2.pt     (porteus_dqn_v2.py)
#   python simulate.py --agent v2 --model other.pt --episodes 5 --show
#
# Each frame shows: cliff (red), bridge (orange), start (blue), goal (green),
# the agent (black dot), its path so far, and the greedy action in every cell
# (arrows). The v2 agent also sees whether the bridge was crossed, so its
# arrows switch to the "after bridge" policy once it crosses.

import argparse
import contextlib
import csv
import io

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import animation, patches

from risky_bridge_cliff import RiskyBridgeCliffWorld

ARROWS = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}  # (dx, dy) in plot coords
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


# ------------------------------------------------------------
# Policies: each returns scores(row, col, bridge_flag) -> array of 4,
# and the greedy action is argmax of the scores.
# ------------------------------------------------------------

def load_tabular(path):
    table = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (int(row["row"]), int(row["col"]))
            table[key] = np.array([float(row[f"z_{a}"]) for a in ACTION_NAMES])

    def scores(state):
        # Greedy is argmin z, i.e. argmax -z. Unseen states default to zeros,
        # as in tabular_porteus_tail.py. Ignores the bridge flag.
        return -table.get(tuple(state[:2]), np.zeros(4))

    return scores


def load_dqn(path, env):
    import torch
    from risk_sensitive_dqn import TailNetwork, normalize_state

    ckpt = torch.load(path, map_location="cpu")
    hidden = ckpt["config"].get("hidden_dim", 128)
    net = TailNetwork(state_dim=2, num_actions=4, hidden_dim=hidden)
    net.load_state_dict(ckpt["online_net"])
    net.eval()

    def scores(state):
        s = normalize_state(state[:2], env.rows, env.cols)
        with torch.no_grad():
            return -net(torch.tensor(s).unsqueeze(0))[0].numpy()  # argmin z

    return scores


def load_v2(path, env):
    import torch
    from porteus_dqn_v2 import QNetwork, make_featurizer

    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    feat, in_dim = make_featurizer(env, cfg.get("features", "onehot"))
    net = QNetwork(in_dim, 4, cfg.get("hidden_dim", 128))
    net.load_state_dict(ckpt["online"])
    net.eval()
    print(f"v2 model: lambda = {cfg.get('risk_lambda')}, k = {cfg.get('stationarity_index')}")

    def scores(state):
        with torch.no_grad():
            return net(torch.tensor(feat(np.array(state))).unsqueeze(0))[0].numpy()

    return scores


# ------------------------------------------------------------
# Roll out episodes, recording every frame
# ------------------------------------------------------------

def full_state(env):
    return (*env.state, int(env.visited_bridge))


def rollout(env, scores, episodes, seed, max_steps):
    frames = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        state = full_state(env)
        path = [state[:2]]
        ret = 0.0
        frames.append(dict(ep=ep, t=0, state=state, path=list(path),
                           ret=ret, reward=None, action=None, event=""))
        for t in range(1, max_steps + 1):
            action = int(np.argmax(scores(state)))
            with contextlib.redirect_stdout(io.StringIO()):  # silence env prints
                _, reward, term, trunc, info = env.step(action)
            state = full_state(env)
            path.append(state[:2])
            ret += reward
            event = info.get("event", "")
            if event == "goal":
                event = f"goal ({info.get('route', '')} route)"
            if trunc and not term:
                event = "time limit"
            frames.append(dict(ep=ep, t=t, state=state, path=list(path),
                               ret=ret, reward=reward, action=action, event=event))
            if term or trunc:
                break
        if not (term or trunc):
            frames[-1]["event"] = f"stopped after {max_steps} steps"
        # Hold the final frame of each episode a little longer.
        frames.extend([frames[-1]] * 4)
    return frames


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------

def draw_board(ax, env):
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)  # row 0 at the top
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    def cell(rc, color, label):
        r, c = rc
        ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=color, edgecolor="grey"))
        if label:
            ax.text(c, r + 0.32, label, ha="center", va="center", fontsize=8)

    for r in range(env.rows):
        for c in range(env.cols):
            cell((r, c), "white", "")
    for rc in env.cliff:
        cell(rc, "#f4a3a3", "cliff")
    cell(env.bridge, "#f7c873", f"bridge\np={env.slip_probability}")
    cell(env.start, "#a9c8f0", "start")
    cell(env.goal, "#a6dba0", "goal")


def draw_arrows(ax, env, scores, flag):
    """Greedy-policy arrows in every non-terminal cell, for a bridge flag."""
    artists = []
    terminal = set(env.cliff) | {env.goal}
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in terminal:
                continue
            a = int(np.argmax(scores((r, c, flag))))
            dx, dy = ARROWS[a]
            artists.append(ax.arrow(c - dx / 2, r - dy / 2, dx, dy,
                                    head_width=0.12, length_includes_head=True,
                                    color="#888888", alpha=0.7))
    return artists


def animate(env, scores, frames, title, out_path, fps, show):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    draw_board(ax, env)
    arrows = {"flag": 0, "artists": draw_arrows(ax, env, scores, 0)}
    (trail,) = ax.plot([], [], "-", color="black", alpha=0.35, lw=3)
    (dot,) = ax.plot([], [], "o", color="black", ms=16)
    text = ax.set_title("")

    def update(i):
        f = frames[i]
        flag = f["state"][2]
        if flag != arrows["flag"]:  # redraw the policy after crossing the bridge
            for art in arrows["artists"]:
                art.remove()
            arrows["artists"] = draw_arrows(ax, env, scores, flag)
            arrows["flag"] = flag
        rows = [p[0] for p in f["path"]]
        cols = [p[1] for p in f["path"]]
        trail.set_data(cols, rows)
        dot.set_data([f["state"][1]], [f["state"][0]])
        last = "" if f["action"] is None else (
            f"action {ACTION_NAMES[f['action']]}, reward {f['reward']:+.0f}")
        bridge = "  (bridge crossed)" if flag else ""
        text.set_text(f"{title}\nepisode {f['ep'] + 1}  step {f['t']}  "
                      f"return {f['ret']:+.0f}   {last}\n{f['event']}{bridge}")
        return trail, dot, text

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    print(f"Saved animation to {out_path} ({len(frames)} frames)")
    if show:
        plt.show()
    plt.close(fig)


DEFAULT_MODELS = {"tabular": "tabular_z_values.csv",
                  "dqn": "porteus_tail_dqn.pt",
                  "v2": "porteus_dqn_v2.pt"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["tabular", "dqn", "v2"], default="tabular")
    p.add_argument("--model", help="path to the saved agent (default depends on --agent)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--slip", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--fps", type=float, default=3)
    p.add_argument("--out", help="output GIF path")
    p.add_argument("--show", action="store_true",
                   help="also open a window (needs a display)")
    args = p.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    env = RiskyBridgeCliffWorld(slip_probability=args.slip,
                                include_bridge_flag=(args.agent == "v2"),
                                verbose=False)
    model = args.model or DEFAULT_MODELS[args.agent]
    if args.agent == "tabular":
        scores = load_tabular(model)
    elif args.agent == "dqn":
        scores = load_dqn(model, env)
    else:
        scores = load_v2(model, env)

    frames = rollout(env, scores, args.episodes, args.seed, args.max_steps)
    out = args.out or f"simulation_{args.agent}.gif"
    animate(env, scores, frames, f"{args.agent} agent (greedy policy)",
            out, args.fps, args.show)


if __name__ == "__main__":
    main()
