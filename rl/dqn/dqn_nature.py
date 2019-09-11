import os
import tensorflow as tf
import numpy as np
import random
from collections import deque
import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../util'))

from gym_wrapper import make_atari
from config import get_option

option = get_option("dqn")

ENV_CONFIG = {
    "env_id": option.env_name,
    "noop": option.noop,
    "skip_frame": option.action_interval,
    "episode_life": option.episode_life,
    "frame_h": option.height,
    "frame_w": option.width,
    "frame_clip": option.frame_clip,
    "clip_rewards": option.clip_rewards,
    "frame_stack": option.state_length,
    "scale": option.scale,
    "monitor": option.monitor,
    "time_limit": option.time_limit}

VARIABLE_ENV_CONFIG = ENV_CONFIG.copy()
VARIABLE_ENV_CONFIG.update({
    "noop": None,
    "monitor": option.variable_monitor,
    "time_limit": option.variable_time_limit})

tf.set_random_seed(0)
np.random.seed(0)
random.seed(0)


class sammarizer(object):
    def __init__(self, sess):
        # Parameters used for result
        self.reset_param()
        self.setup_summary()
        self.sess = sess
        self.summary_writer = tf.summary.FileWriter(option.save_summary_path, self.sess.graph)

    def setup_summary(self):
        with tf.name_scope(str(option.env_name)):
            episode_total_reward = tf.Variable(0., name="Reward", dtype=tf.float32)
            tf.summary.scalar('/Total Reward/Episode', episode_total_reward)
            episode_avg_max_q = tf.Variable(0., name="Q", dtype=tf.float32)
            tf.summary.scalar('/Average Max Q/Episode', episode_avg_max_q)
            episode_duration = tf.Variable(0., name="Duration", dtype=tf.float32)
            tf.summary.scalar('/Duration/Episode', episode_duration)
            episode_avg_loss = tf.Variable(0., name="Loss", dtype=tf.float32)
            tf.summary.scalar('/Average Loss/Episode', episode_avg_loss)

            valuation = tf.Variable(0., name="Valuation", dtype=tf.float32)
            self.valuation_ph = tf.placeholder(tf.float32)
            self.valuation_update_op = valuation.assign(self.valuation_ph)
            tf.summary.scalar('/Valuation', valuation, collections=["valuation"])
            self.summary_valuation_op = tf.summary.merge_all(key="valuation")

            summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]

            self.summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
            self.update_ops = [summary_vars[i].assign(self.summary_placeholders[i]) for i in range(len(summary_vars))]
            self.summary_op = tf.summary.merge_all()

    def reset_param(self):
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0

    def step(self, reward, q_v, loss):
        self.total_reward += reward
        self.total_q_max += q_v
        self.total_loss += loss
        pass

    def write(self, total_step, episode, duration, epsilon):
        stats = [self.total_reward, self.total_q_max / float(duration),
                 duration, self.total_loss / (float(duration) / float(option.action_interval))]
        for i in range(len(stats)):
            self.sess.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])})
        summary_str = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, total_step)

        print(
            'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f}'.format(
                episode + 1, total_step, duration, epsilon,
                self.total_reward, self.total_q_max / float(duration),
                self.total_loss / (float(duration) / float(option.action_interval)),
            ))
        self.reset_param()

    def val_write(self, total_step, reword):
        print("val reword: {}".format(reword))
        self.sess.run(self.valuation_update_op, feed_dict={self.valuation_ph: reword})
        summary_str = self.sess.run(self.summary_valuation_op)
        self.summary_writer.add_summary(summary_str, total_step)


class Agent(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.t = 0

        # Create replay memory
        self.replay_memory = deque(maxlen=option.num_replay_memory)

        # Create q network
        self.s, self.q_values, self.q_network = self.build_network(name="q-net")

        self.st, self.target_q_values, self.target_q_network = self.build_network(name="target-net")

        # Define loss and gradient update operation
        self.a, self.terminal, self.reward, self.loss, self.grad_update = self.build_training_op(self.q_values,
                                                                                                 self.target_q_values)

        self.update_target_data, self.update_target_network, self.update_target_ph = self.build_net_update_op(
            self.target_q_network)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        if not os.path.exists(option.save_network_path):
            os.makedirs(option.save_network_path)
        pd.DataFrame().from_dict(option.__dict__, orient='index').to_csv(option.save_network_path + "/config.csv")
        self.sess.run(tf.initialize_all_variables())

        # Load network
        if option.load:
            self.load_network()

    def _build_cnn(self, input, name, shape, stride, actibation=tf.nn.relu, initializer=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                w = tf.get_variable("weight_{}".format(name), shape)
                b = tf.get_variable("b_{}".format(name), [shape[-1]])
            else:
                w = tf.get_variable("weight_{}".format(name), shape, initializer=initializer, dtype=tf.float32)
                b = tf.get_variable("b_{}".format(name), [shape[-1]], initializer=initializer, dtype=tf.float32)

            return actibation(tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='VALID') + b)

    def _build_fl(self, input, name, output, actibation=tf.nn.relu, initializer=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                w = tf.get_variable("weight_{}".format(name), [input.get_shape()[1], output])
                b = tf.get_variable("b_{}".format(name), [output])
            else:
                w = tf.get_variable("weight_{}".format(name), [input.get_shape()[1], output],
                                    initializer=initializer, dtype=tf.float32)
                b = tf.get_variable("b_{}".format(name), [output], initializer=initializer, dtype=tf.float32)

            if actibation is None:
                return tf.add(tf.matmul(input, w), b)
            else:
                return actibation(tf.add(tf.matmul(input, w), b))

    def build_network(self, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            s = tf.placeholder(tf.float32, [None, option.state_length, option.width, option.height])
            trans_s = tf.transpose(s, [0, 2, 3, 1])
            weight_init = tf.truncated_normal_initializer(stddev=0.01)

            conv1 = self._build_cnn(trans_s, "conv1", [8, 8, 4, 32], 4, initializer=weight_init, reuse=reuse)
            conv2 = self._build_cnn(conv1, "conv2", [4, 4, 32, 64], 2, initializer=weight_init, reuse=reuse)
            conv3 = self._build_cnn(conv2, "conv3", [3, 3, 64, 64], 1, initializer=weight_init, reuse=reuse)
            h_fratten = tf.layers.Flatten()(conv3)
            fl4 = self._build_fl(h_fratten, "fl4", 512, initializer=weight_init, reuse=reuse)
            output = self._build_fl(fl4, "fl5", env.action_space.n, actibation=None, initializer=weight_init,
                                    reuse=reuse)

            # temp=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return s, output, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def build_training_op(self, q_values, qt_values):
        with tf.variable_scope('train'):
            a = tf.placeholder(tf.uint8, [None])
            terminal = tf.placeholder(tf.float32, [None])
            reward = tf.placeholder(tf.float32, [None])

            qt_v = tf.stop_gradient(reward + (1 - terminal) * option.gamma * tf.reduce_max(qt_values, 1))

            a_one_hot = tf.one_hot(a, env.action_space.n, 1.0, 0.0)
            a_one_hot = tf.stop_gradient(a_one_hot)
            q_v = tf.reduce_sum(tf.multiply(q_values, a_one_hot), reduction_indices=1)

            loss = tf.losses.huber_loss(qt_v, q_v)

            grad_update = tf.train.RMSPropOptimizer(learning_rate=option.learning_rate, momentum=option.momentum,
                                                    epsilon=option.epsilon).minimize(loss)

        return a, terminal, reward, loss, grad_update

    def build_net_update_op(self, target_q_network):
        update_target_data = {"total_reward": -9999, "network": None}
        update_target_network = []
        update_target_ph = []
        for target in target_q_network:
            ph = tf.placeholder(tf.float32, target.get_shape().as_list())
            update_target_network.append(tf.assign(target, ph))
            update_target_ph.append(ph)
        return update_target_data, update_target_network, update_target_ph

    def run(self, state, next_state, action, reward, terminal, step):
        # Store transition in replay memory
        self.push_replay_memory(state, next_state, action, reward, terminal)

        loss = self.train_network()

        if step % option.target_update_interval == 0:
            self.update_network()
        return loss

    def check_update_network(self, total_reword):
        if self.update_target_data["total_reward"] < total_reword:
            self.update_target_data["total_reward"] = total_reword
            self.update_target_data["network"] = self.sess.run(self.q_network)
        pass

    def update_network(self):
        print("update start")
        feed_dic = {}
        for ph, v in zip(self.update_target_ph, self.update_target_data["network"]):
            feed_dic[ph] = v
        self.sess.run(self.update_target_network, feed_dict=feed_dic)
        print("reword:{}".format(self.update_target_data["total_reward"]))
        self.update_target_data["total_reward"] = -9999
        print("update end")

    def save_network(self, step):
        save_path = self.saver.save(self.sess, option.save_network_path, global_step=(step))
        print('Successfully saved: ' + save_path)

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, option.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            next_state_batch.append(data[1])
            action_batch.append(data[2])
            reward_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        state_batch = np.array(state_batch, dtype=np.float32) / self.maxv
        next_state_batch = np.array(next_state_batch, dtype=np.float32) / self.maxv
        action_batch = np.array(action_batch, dtype=np.uint8)
        reward_batch = np.array(reward_batch, dtype=np.float32)
        terminal_batch = np.array(terminal_batch, dtype=np.float32)

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: state_batch,
            self.st: next_state_batch,
            self.a: action_batch,
            self.reward: reward_batch,
            self.terminal: terminal_batch,
        })

        return loss

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(option.save_network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action(self, state, epsilon=0.05):
        q_v = self.sess.run(self.q_values, feed_dict={self.s: np.array([state / self.maxv], dtype=np.float32)})
        if epsilon >= random.random():
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(q_v)
        return action, np.max(q_v)

    def push_replay_memory(self, state, next_state, action, reward, terminal):
        self.replay_memory.append((state, next_state, action, reward, terminal))


def worm_up(agent, env):
    state = env.reset()
    agent.maxv = np.max(state)
    for episode in range(option.initial_replay_size):
        action, _ = agent.get_action(state, epsilon=1.0)
        next_state, reward, terminal, _ = env.step(action)
        agent.push_replay_memory(state, next_state, action, reward, terminal)
        if terminal:
            state = env.reset()
        else:
            state = next_state


if __name__ == '__main__':

    env = make_atari(**ENV_CONFIG)
    val_env = make_atari(**VARIABLE_ENV_CONFIG)

    agent = Agent(num_actions=env.action_space.n)

    if option.train:  # Train mode

        env.seed(0)

        total_step = 0
        worm_up(agent, env)
        epsilon = option.initial_epsilon
        epsilon_step = (option.initial_epsilon - option.final_epsilon) / option.exploration_steps

        summary = sammarizer(agent.sess)
        agent.check_update_network(total_reword=0)

        for episode in range(option.num_episodes):
            duration = 0
            terminal = False
            state = env.reset()
            agent.maxv = np.max(state)
            while not terminal:
                # Anneal epsilon linearly over time
                if epsilon > option.final_epsilon:
                    epsilon -= epsilon_step

                action, q_v = agent.get_action(state, epsilon)
                next_state, reward, terminal, _ = env.step(action)
                loss = agent.run(state, next_state, action, reward, terminal, total_step)
                state = next_state

                if total_step % option.valuation_interval == 0:
                    print("start valuation")
                    val_reword = 0
                    for _ in range(option.valuation_num):
                        terminal = False
                        state = val_env.reset()
                        agent.maxv = np.max(state)
                        while not terminal:
                            action, _ = agent.get_action(state)
                            next_state, reward, terminal, _ = val_env.step(action)
                            state = next_state
                            val_reword += reward
                    summary.val_write(total_step, val_reword / option.valuation_num)
                    print("end valuation")

                summary.step(reward, q_v, loss)
                total_step += 1
                duration += 1

            agent.check_update_network(total_reword=summary.total_reward)
            summary.write(total_step, episode, duration, epsilon)

    else:  # Test mode
        for _ in range(option.num_episodes_at_test):
            terminal = False
            state = val_env.reset()
            agent.maxv = np.max(state)
            while not terminal:
                action, _ = agent.get_action(state)
                next_state, reward, terminal, _ = val_env.step(action)
                state = next_state
                val_env.render()
