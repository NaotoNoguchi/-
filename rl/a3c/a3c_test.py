# import better_exceptions
import os
import tensorflow as tf
import gym
from gym import wrappers

from network import *
from async_agent import *
from gym_wrapper import make_atari
flags = tf.app.flags

flags.DEFINE_string('device', 'cpu', '[GPU,CPU]')
flags.DEFINE_string('game', 'BreakoutNoFrameskip-v4', 'The type of game environment you want to ')
flags.DEFINE_string('out', '/tmp/result', 'Specify a output directory for your run')
flags.DEFINE_string('model', './log/SpaceInvadersNoFrameskip-v4_lstmadam_02142149/model.ckpt-580000', 'Specify a pretrained model file')

flags.DEFINE_integer('iter', 100, 'The number of games you want to run')
flags.DEFINE_string('policy', 'greedy', '[greedy,softmax] policy')
flags.DEFINE_boolean('render', False, 'render it or not')

assert flags.FLAGS.device in ['gpu', 'cpu'], 'Device should be either "GPU" or "CPU"'
assert flags.FLAGS.policy in ['greedy', 'softmax'], 'Device should be either "greedy" or "softmax"'






ENV_CONFIG = {
    "env_id": "SpaceInvadersNoFrameskip-v4",
    # "env_id": 'MsPacmanNoFrameskip-v4',
    # "env_id": 'AtlantisNoFrameskip-v4',
    # "env_id": str(None),
    # "env_id": 'BreakoutNoFrameskip-v4',
    "noop": 30,
    "skip_frame": 3,
    "episode_life": False,
    "frame_h": 84,
    "frame_w": 84,
    "frame_clip": False,
    "clip_rewards": False,
    "frame_stack": 4,
    "scale": True,
    "monitor": None,
    "time_limit": None}




sample_env = wrappers.Monitor(make_atari(**ENV_CONFIG), os.path.join(flags.FLAGS.out), force=True)

nA = sample_env.action_space.n

device = '/gpu:0' if (flags.FLAGS.device == 'gpu') else '/cpu:0'
ac = ActorCritic(nA, device_name=device,
                 learning_rate=None, decay=None, grad_clip=None,
                 entropy_beta=None)
agent = A3CGroupAgent([sample_env], ac, unroll_step=0, discount_factor=0.)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=ac.train_vars)

sess = tf.Session()
sess.graph.finalize()

sess.run(init_op)
agent.ac.initialize(sess)
saver.restore(sess, flags.FLAGS.model)

greedy = True if (flags.FLAGS.policy == 'greedy') else False
for _ in range(flags.FLAGS.iter):
    print(agent.test_run(sample_env, greedy=greedy, render=flags.FLAGS.render))

sess.close()
