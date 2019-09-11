from argparse import ArgumentParser
from datetime import datetime

TIME_STAMP = datetime.now().strftime("%Y%m%d%H%M")


def get_option(model_name):
    argparser = ArgumentParser()

    argparser.add_argument('-si', '--save_interval', type=int,
                           default=300000,
                           help='The frequency with which the network is saved')

    # learning config
    argparser.add_argument('-d', '--device', type=str,
                           default="/cpu:0", help='device')
    argparser.add_argument('-t', '--train', type=bool,
                           default=True, help='is train')
    argparser.add_argument('-l', '--load',
                           default=None, help='load model')
    argparser.add_argument('-ne', '--num_episodes', type=int,
                           default=1000000,
                           help='Number of episodes the agent plays')

    argparser.add_argument('-snp', '--save_network_path',
                           default=None,
                           help='save network path')
    argparser.add_argument('-ssp', '--save_summary_path',
                           default=None,
                           help='save summary path')

    argparser.add_argument('-vi', '--valuation_interval', type=int,
                           default=10000,
                           help='Number of valuation interval')
    argparser.add_argument('-vn', '--valuation_num', type=int,
                           default=5,
                           help='Number of valuation num')

    argparser.add_argument('-net', '--num_episodes_at_test', type=int,
                           default=1,
                           help='Number of episodes the agent plays at test time')

    # common model config
    argparser.add_argument('-W', '--width', type=int,
                           default=84, help='Resized frame width')
    argparser.add_argument('-H', '--height', type=int,
                           default=84, help='Resized frame height')
    argparser.add_argument('-sl', '--state_length', type=int,
                           default=4,
                           help='Number of most recent frames to produce the input to the network')

    argparser.add_argument('-bs', '--batch_size', type=int,
                           default=32,
                           help='Mini batch size')

    if "a3c" in model_name:
        # A3C setting
        argparser.add_argument('-lr', '--learning_rate', type=float,
                               default=0.00025,
                               help='optimizer learning rate')
        argparser.add_argument('-df', '--discount_factor', type=float,
                               default=0.99,
                               help='discount factor')

        argparser.add_argument('-dc', '--decay', type=float,
                               default=0.99,
                               help='decay')
        argparser.add_argument('-gc', '--grad_clip', type=float,
                               default=0.1,
                               help='gradient clip')
        argparser.add_argument('-eb', '--entropy_beta', type=float,
                               default=0.01,
                               help='entropy beta')

        argparser.add_argument('-nt', '--num_threads', type=int,
                               default=4,
                               help='num of threads')
        argparser.add_argument('-an', '--agent_per_threads', type=int,
                               default=32,
                               help='num of agent per threads')
        argparser.add_argument('-us', '--unroll_step', type=int,
                               default=5,
                               help='unrolsl steps')
        argparser.add_argument('-sp', '--summary_period', type=int,
                               default=100,
                               help='summary period')
        argparser.add_argument('-sep', '--save_period', type=int,
                               default=20000,
                               help='save_period')
    if "dqn" in model_name:
        # DQN setting
        argparser.add_argument('-es', '--exploration_steps', type=int,
                               default=1000000,
                               help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
        argparser.add_argument('-nrm', '--num_replay_memory', type=int,
                               default=600000,
                               help='Number of replay memory the agent uses for training')
        argparser.add_argument('-irs', '--initial_replay_size', type=int,
                               default=10000,
                               help='Number of steps to populate the replay memory before training starts')

        argparser.add_argument('-lr', '--learning_rate', type=float,
                               default=0.001,
                               help='optimizer learning rate')
        argparser.add_argument('-mm', '--momentum', type=float,
                               default=0.95,
                               help='optimizer momentum')
        argparser.add_argument('-ep', '--epsilon', type=float,
                               default=0.01,
                               help='optimizer epsilon')

        argparser.add_argument('-g', '--gamma', type=float,
                               default=0.99,
                               help='Discount factor')
        argparser.add_argument('-ie', '--initial_epsilon', type=float,
                               default=1.0,
                               help='Initial value of epsilon in epsilon-greedy')
        argparser.add_argument('-fe', '--final_epsilon', type=float,
                               default=0.99,
                               help=' Final value of epsilon in epsilon-greedy')

        argparser.add_argument('-tui', '--target_update_interval', type=int,
                               default=10000,
                               help='The frequency with which the target network is updated')

    # Game setting
    argparser.add_argument('-ai', '--action_interval', type=int,
                           default=4,
                           help='The frequency with which the target network is updated')
    argparser.add_argument('-en', '--env_name',
                           default=None,
                           help='Environment name')
    # "env_name": "SpaceInvadersNoFrameskip-v4",
    # "env_name": 'MsPacmanNoFrameskip-v4',
    # "env_name": 'AtlantisNoFrameskip-v4',
    # "env_name": 'BreakoutNoFrameskip-v4',

    argparser.add_argument('-no', '--noop', type=int,
                           default=30)
    argparser.add_argument('-el', '--episode_life', type=bool,
                           default=False)
    argparser.add_argument('-fc', '--frame_clip', type=bool,
                           default=True)
    argparser.add_argument('-cr', '--clip_rewards', type=bool,
                           default=True)
    argparser.add_argument('-sc', '--scale', type=bool,
                           default=False)
    argparser.add_argument('-mo', '--monitor',
                           default=None)
    argparser.add_argument('-tl', '--time_limit',
                           default=None)
    argparser.add_argument('-vmo', '--variable_monitor',
                           default=None)
    argparser.add_argument('-vtl', '--variable_time_limit',
                           default=None)

    option = argparser.parse_args()
    if not option.save_summary_path:
        option.save_summary_path = "../result/{}_{}_{}".format(option.env_name, model_name, TIME_STAMP)
    else:
        option.save_summary_path = "{}/{}_{}_{}".format(option.save_network_path, option.env_name, model_name,
                                                        TIME_STAMP)

    if not option.save_network_path:
        option.save_network_path = "../result/{}_{}_{}".format(option.env_name, model_name, TIME_STAMP)
    else:
        option.save_network_path = "{}/{}_{}_{}".format(option.save_network_path, option.env_name, model_name,
                                                        TIME_STAMP)

    return option
