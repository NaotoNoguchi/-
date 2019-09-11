from collections import deque

import numpy as np
import gym
from gym import spaces
from gym.wrappers import Monitor, TimeLimit
from TestEnv import TestEnv


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


import cv2


class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, height, width, frame_clip=False, interpolation="INTER_LINEAR"):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.height = height
        self.width = width
        if frame_clip:
            self.resize_func = self.clip_resize
        else:
            self.resize_func = self.resize
            # self.resize_func = self.clip_resize

        if interpolation == "INTER_NEAREST":  # 最近傍補間
            self.interpolation = cv2.INTER_NEAREST
        elif interpolation == "INTER_LINEAR":  # バイリニア補間（デフォルト）
            self.interpolation = cv2.INTER_LINEAR
        elif interpolation == "INTER_CUBIC":  # 4x4 の近傍領域を利用するバイキュービック補間
            self.interpolation = cv2.INTER_CUBIC
        elif interpolation == "INTER_AREA":  # 領域の関係を利用したリサンプリング．画像を大幅に縮小する場合は，モアレを避けることができる良い手法です
            self.interpolation = cv2.INTER_AREA
        elif interpolation == "INTER_LANCZOS4":  # 8x8 の近傍領域を利用する Lanczos法の補間
            self.interpolation = cv2.INTER_LANCZOS4
        else:
            self.interpolation = None

        self.frame_clip = frame_clip
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

    def _observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = self.resize_func(frame)
        return np.uint8(frame)

    def clip_resize(self, frame):
        frame = cv2.resize(frame, (self.width, 110), interpolation=self.interpolation)
        return np.array(frame[18:102, :])

    def resize(self, frame):
        return cv2.resize(frame, (self.width, self.height), interpolation=self.interpolation)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return list(self.frames)

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return list(self.frames), reward, done, info


class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def make_atari(env_id=None, noop=30, skip_frame=4, episode_life=True, frame_h=84, frame_w=84, frame_clip=False,
               clip_rewards=True, frame_stack=4, scale=False, monitor=300000, time_limit=300):
    """Configure environment for DeepMind-style Atari.
    """
    if env_id is not None:
        env = gym.make(env_id)
    else:
        env = TestEnv()
    assert 'NoFrameskip' in env.spec.id
    if noop is not None:
        env = NoopResetEnv(env, noop_max=noop)
    if skip_frame is not None and skip_frame != 0:
        env = SkipEnv(env, skip=skip_frame)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, height=frame_h, width=frame_w, frame_clip=(env_id is not None and frame_clip))
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack is not None:
        env = FrameStack(env, frame_stack)
    if monitor is not None:
        env = Monitor(env, "./movie", video_callable=(lambda ep: ep % monitor == 0))
    if time_limit is not None:
        env = TimeLimit(env, max_episode_seconds=time_limit)

    return env


from TestEnv import TestEnv

if __name__ == '__main__':
    env = TestEnv()
    env = make_atari(env=env, noop=False, skip_frame=True, episode_life=False, clip_rewards=True,
                     frame_stack=True, scale=False)
    # env = make_atari("CartPole-v0")

    _ = env.reset()

    for i in range(10):
        _ = env.step(0)
        pass
