from tqdm import tqdm
from network import *
from async_agent import *
import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../util'))

from gym_wrapper import make_atari
from config import get_option

option = get_option("a3c")


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

if __name__ == "__main__":
    tf.set_random_seed(0)
    np.random.seed(0)

    if not os.path.exists(option.save_network_path):
        os.makedirs(option.save_network_path)
    pd.DataFrame().from_dict(option.__dict__, orient='index').to_csv(option.save_network_path + "/config.csv")

    tf.reset_default_graph()
    sample_env = make_atari(**ENV_CONFIG)

    nA = sample_env.action_space.n

    # define actor critic networks and environments
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(option.learning_rate, global_step, option.num_episodes // 2,
                                              option.learning_rate * 0.1)
    master_ac = ActorCritic(nA, device_name=option.device,
                            learning_rate=learning_rate, decay=option.decay, grad_clip=option.grad_clip,
                            entropy_beta=option.entropy_beta)
    group_agents = [
        A3CGroupAgent([make_atari(**ENV_CONFIG) for _ in range(option.agent_per_threads)],
                      ActorCritic(nA, master=master_ac, device_name=option.device, scope_name='Thread%02d' % i,
                                  learning_rate=learning_rate, decay=option.decay, grad_clip=option.grad_clip,
                                  entropy_beta=option.entropy_beta),
                      unroll_step=option.unroll_step,
                      discount_factor=option.discount_factor,
                      seed=i)
        for i in range(option.num_threads)]

    queue = tf.FIFOQueue(capacity=option.num_threads * 10,
                         dtypes=[tf.float32, tf.float32, tf.float32], )
    qr = tf.train.QueueRunner(queue, [g_agent.enqueue_op(queue) for g_agent in group_agents])
    tf.train.queue_runner.add_queue_runner(qr)
    loss = queue.dequeue()

    # Miscellaneous(init op, summaries, etc.)
    increase_step = global_step.assign(global_step + 1)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]


    def _train_info():
        total_eps = sum([g_agent.num_episodes() for g_agent in group_agents])
        avg_r = sum([g_agent.reward_info()[0] for g_agent in group_agents]) / len(group_agents)
        max_r = max([g_agent.reward_info()[1] for g_agent in group_agents])
        return total_eps, avg_r, max_r


    train_info = tf.py_func(_train_info, [], [tf.int64, tf.float64, tf.float64], stateful=True)
    pl, el, vl = loss
    total_eps, avg_r, max_r = train_info

    tf.summary.scalar('/{}/learning_rate'.format(ENV_CONFIG["env_id"]), learning_rate)
    tf.summary.scalar('/{}/policy_loss'.format(ENV_CONFIG["env_id"]), pl)
    tf.summary.scalar('/{}/entropy_loss'.format(ENV_CONFIG["env_id"]), el)
    tf.summary.scalar('/{}/value_loss'.format(ENV_CONFIG["env_id"]), vl)
    tf.summary.scalar('/{}/total_episodes'.format(ENV_CONFIG["env_id"]), total_eps)
    tf.summary.scalar('/{}/average_rewards'.format(ENV_CONFIG["env_id"]), avg_r)
    tf.summary.scalar('/{}/maximum_rewards'.format(ENV_CONFIG["env_id"]), max_r)
    summary_op = tf.summary.merge_all()
    # config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

    # saver and sessions
    saver = tf.train.Saver(var_list=master_ac.train_vars, max_to_keep=5)

    sess = tf.Session()
    sess.graph.finalize()

    sess.run(init_op)
    master_ac.initialize(sess)
    for agent in group_agents:
        agent.ac.initialize(sess)
    print('Initialize Complete...')

    try:
        summary_writer = tf.summary.FileWriter(option.save_summary_path, sess.graph)
        summary_writer_eps = tf.summary.FileWriter(os.path.join(option.save_summary_path, 'per-eps'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in tqdm(range(option.num_episodes)):
            if coord.should_stop():
                break

            (pl, el, vl), summary_str, (total_eps, avg_r, max_r), _ = sess.run(
                [loss, summary_op, train_info, increase_step])
            if (step % option.summary_period == 0):
                summary_writer.add_summary(summary_str, step)
                summary_writer_eps.add_summary(summary_str, total_eps)
                tqdm.write(
                    'step(%7d) policy_loss:%1.5f,entropy_loss:%1.5f,value_loss:%1.5f, te:%5d avg_r:%2.1f max_r:%2.1f' %
                    (step, pl, el, vl, total_eps, avg_r, max_r))

            if ((step + 1) % option.save_period == 0):
                saver.save(sess, option.save_network_path + '/model.ckpt', global_step=step + 1)
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

        saver.save(sess, option.save_network_path + '/last.ckpt')
        sess.close()

        # queue.close() #where should it go?
