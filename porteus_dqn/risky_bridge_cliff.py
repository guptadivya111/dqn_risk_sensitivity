
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class RiskyBridgeCliffWorld(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, slip_probability=0.1, include_bridge_flag=False,
                 verbose=True):
        # include_bridge_flag: observation becomes (row, col, visited_bridge).
        #   The goal reward (20 vs 6) depends on visited_bridge, so without
        #   this flag the observed state is not Markov.
        # verbose: print "Goal reached" when the goal is reached.

        super().__init__()

        self.include_bridge_flag = include_bridge_flag
        self.verbose = verbose

        self.rows = 5
        self.cols = 7

        self.start = (4, 0)
        self.goal = (4, 6)
        self.bridge = (3, 3)

        self.cliff = [
            (4, 1), (4, 2), (4, 3),
            (4, 4), (4, 5)
        ]

        self.slip_probability = slip_probability

        assert 0 <= slip_probability <= 1

        # State: row * 7 + column
        if include_bridge_flag:
            self.observation_space = spaces.MultiDiscrete([5, 7, 2])
        else:
            self.observation_space = spaces.MultiDiscrete([5, 7])

        # Actions: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        self.moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        self.state = self.start
        self.visited_bridge = False
        self.time = 0

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = self.start
        self.visited_bridge = False
        self.time = 0

        return self._obs(), {}

    def _obs(self):
        if self.include_bridge_flag:
            return np.array((*self.state, int(self.visited_bridge)),
                            dtype=np.int64)
        return np.array(self.state, dtype=np.int64)

    def step(self, action):

        self.time += 1 
        
        dr, dc = self.moves[int(action)]

        r, c = self.state

        next_state = (r + dr, c + dc)

        # Boundary checking
        nr, nc = next_state

        if not (
            0 <= nr < self.rows
            and 0 <= nc < self.cols
        ):
            next_state = self.state

        reward = -1.0
        terminated = False
        truncated = False
        info = {}

        # Stochastic bridge
        if next_state == self.bridge:

            if self.np_random.random() < self.slip_probability:

                # Fall into the cliff
                next_state = (4, 3)

                reward = -100.0
                terminated = True

                info["event"] = "bridge_failure"

            else:

                self.visited_bridge = True

        # Check for cliff
        if next_state in self.cliff:

            reward = -100.0
            terminated = True

            info["event"] = "cliff"

        # Check for goal
        elif next_state == self.goal:

            if self.visited_bridge:
                reward = 20.0
                info["route"] = "risky"

            else:
                reward = 6.0
                info["route"] = "safe"

            terminated = True
            if self.verbose:
                print('Goal reached')
                print('Goal reached')
                print('Goal reached')
            info["event"] = "goal"

        if self.time >=200:
            truncated = True
            
        self.state = next_state

        return (
            self._obs(),
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):

        grid = np.full(
            (self.rows, self.cols),
            ".",
            dtype="<U2"
        )

        for r, c in self.cliff:
            grid[r, c] = "C"

        grid[self.bridge] = "B"
        grid[self.start] = "S"
        grid[self.goal] = "G"

        grid[self.state] = "A"

        print("\n" + "-" * 25)

        for row in grid:
            print(" ".join(row))

        print("-" * 25)

        print("A = Agent")
        print("S = Start")
        print("G = Goal")
        print("B = Risky bridge")
        print("C = Cliff")
