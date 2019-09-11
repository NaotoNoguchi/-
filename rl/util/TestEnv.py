from gym import error, spaces, Env
import os
import numpy as np
import hashlib

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}


class TestEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class ale:
        @classmethod
        def lives(cls):
            return 0

    class spec:
        id = "NoFrameskip"

    def __init__(self):
        # parameters
        self._action_set = [0, 3, 4]
        self.action_space = spaces.Discrete(len(self._action_set))
        self.name = os.path.splitext(os.path.basename(__file__))[0]

        self.screen_n_cols = 30
        self.screen_n_rows = 30
        self.player_length = 3
        self.enable_actions = (0, 1, 2)
        self.np_random = np.random
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_n_cols, self.screen_n_rows, 3))

        # variables
        self.reset()

    def step(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_col = max(0, self.player_col - 1)
        elif action == self.enable_actions[2]:
            # move right
            self.player_col = min(self.player_col + 1, self.screen_n_cols - self.player_length)
        else:
            # do nothing
            pass

        # update ball position
        self.ball_row += 1

        # collision detection
        self.reward = 0
        # self.terminal = False
        if self.ball_row == self.screen_n_rows - 1:
            # self.terminal = True
            if self.player_col <= self.ball_col < self.player_col + self.player_length:
                # catch
                self.reward = 1
            else:
                # drop
                self.reward = -1

        self.timer += 1
        self.terminal = self.timer > 300

        screen = self.observe()

        if self.ball_row == self.screen_n_rows - 1:
            self.ball_row = 0
            self.ball_col = np.random.randint(self.screen_n_cols)

        return screen, self.reward, self.terminal, None

    def draw(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols, 3), dtype=np.uint8)
        # self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols, 3))

        # draw player
        self.screen[self.player_row, self.player_col:self.player_col + self.player_length, :] = 100

        # draw ball
        self.screen[self.ball_row, self.ball_col, :] = 100

    def observe(self):
        self.draw()
        return self.screen

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset player position
        self.player_row = self.screen_n_rows - 1
        self.player_col = np.random.randint(self.screen_n_cols - self.player_length)

        # reset ball position
        self.ball_row = 0
        self.ball_col = np.random.randint(self.screen_n_cols)

        # reset other variables
        self.reward = 0
        self.terminal = False

        self.timer = 0

        screen = self.observe()
        return screen

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def render(self, mode='human'):
        img = self.screen
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
