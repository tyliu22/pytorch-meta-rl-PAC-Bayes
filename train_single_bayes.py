import gym
import torch
import json
import os
import yaml
from tqdm import trange
from copy import deepcopy
from functools import reduce
from operator import mul

import maml_rl.envs
from maml_rl.metalearners.maml_rl_train_single_bayes import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers.train_single_bayes_task_sampler import MultiTaskSampler
from maml_rl.utils.helpers_bayes import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args, prior_policy=None, init_from_prior=True, verbose=1):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    args.output_size = reduce(mul, env.action_space.shape, 1)
    args.input_size = reduce(mul, env.observation_space.shape, 1)

    # # Policy
    # policy = get_policy_for_env(args,
    #                             env,
    #                             hidden_sizes=config['hidden-sizes'],
    #                             nonlinearity=config['nonlinearity'])


    # get model:
    # 如果有先验模型，则直接导入先验模型
    if prior_policy and init_from_prior:
        # init from prior model:
        # deepcopy函数：复制并作为一个单独的个体存在；copy函数：复制原有对象，随着原有对象改变而改变
        post_policy = deepcopy(prior_policy).to(args.device)
    else:
        # 否则直接加载新模型
        post_policy = get_policy_for_env(args,
                                         env,
                                         hidden_sizes=config['hidden-sizes'])

    # 数据无需拷贝，即可使用
    # post_policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Unpack parameters:
    # 提取参数
    # optim_func, optim_args, lr_schedule = \
    #     args.optim_func, args.optim_args, args.lr_schedule

    #  Get optimizer:
    # 设置待优化参数
    # optimizer = args.optim_func(post_policy.parameters(), **args.optim_args)

    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               num_tasks=config['meta-batch-size'],
                               policy=post_policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed)

    # tasks['goal'] fast-batch-size = 20 个目标值 0-19
    # 提取batch中的任务： tasks 为 goal，2D任务中的目标值
    # tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
    tasks = sampler.sample_tasks_return()

    # for index, task in enumerate(tasks):
    loss_train = []
    loss_test = []
    for index, task in enumerate(tasks):
        # 针对每一个 task， 采样 fast-batch-size 个 trajectories，
        for batch in trange(config['num-batches']):
            train_episodes, train_loss, valid_episodes, valid_loss = sampler.sample(
                                                            task,
                                                            num_steps=config['num-steps'],
                                                            fast_lr=config['fast-lr'],
                                                            gamma=config['gamma'],
                                                            gae_lambda=config['gae-lambda'],
                                                            device=args.device)
            loss_train.append(train_loss)
            loss_test.append(valid_loss)
    # metalearner = MAMLTRPO(args,
    #                        post_policy,
    #                        fast_lr=config['fast-lr'],
    #                        first_order=config['first-order'])

    num_iterations = 0

"""
    # 训练的次数 num-batches 个 batch
    # 首先针对不同个任务 tasks
    for index, task in enumerate(tasks):
        # 每一个任务对应的 num-batches 个数据
        for batch in trange(config['num-batches']):

        # post_policy.train()
        # post_model.set_eps_std(0.0) # DEBUG: turn off randomness

        # params_show_train = policy.state_dict()

        # tasks['goal'] fast-batch-size = 20 个目标值 0-19
        # 提取batch中的任务： tasks 为 goal，2D任务中的目标值
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])

        # futures = (train_episodes_futures, valid_episodes_futures)
        # 开始采样轨迹
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        # logs = metalearner.step(*futures,
        #                         max_kl=config['max-kl'],
        #                         cg_iters=config['cg-iters'],
        #                         cg_damping=config['cg-damping'],
        #                         ls_max_steps=config['ls-max-steps'],
        #                         ls_backtrack_ratio=config['ls-backtrack-ratio'])
        #
        # train_episodes, valid_episodes = sampler.sample_wait(futures)
        # num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        # num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        # # 更新字典变量 logs 的键值与值
        # logs.update(tasks=tasks,
        #             num_iterations=num_iterations,
        #             train_returns=get_returns(train_episodes[0]),
        #             valid_returns=get_returns(valid_episodes))

        # # Save policy
        # if args.output_folder is not None:
        #     with open(policy_filename, 'wb') as f:
        #         # 保存网络中的参数，f 为路径
        #         torch.save(post_policy.state_dict(), f)
"""

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). '
             'WARNING: Full upport for cuda is not guaranteed. Using CPU is encouraged.'),
    # misc.add_argument('--lr', type=float, help='learning rate (initial)',
    #                     default=1e-3)

    args = parser.parse_args()
    # args.device = ('cuda' if (torch.cuda.is_available()
    #                and args.use_cuda) else 'cpu')

    args.log_var_init = {'mean': -10, 'std': 0.1}  # The initial value for the log-var parameter (rho) of each weight

    # Number of Monte-Carlo iterations (for re-parametrization trick):
    args.n_MC = 1

    args.lr = 1e-3

    args.device = ('cuda' if (torch.cuda.is_available()
                         and args.use_cuda) else 'cpu')
    #  Define optimizer:
    # args.optim_func, args.optim_args = optim.Adam, {'lr': args.lr}
    # prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

    # Learning rate decay schedule:
    # prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10, 30]}
    args.lr_schedule = {}  # No decay

    # 是否从利用先验模型进行初始化
    args.init_from_prior = True

    # Test type:
    args.test_type = 'MaxPosterior'  # 'MaxPosterior' / 'MajorityVote'

    main(args)
