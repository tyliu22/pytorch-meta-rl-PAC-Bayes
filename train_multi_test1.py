"""
************************************************************************************************************************
          multi-task Meta RL based on PAC-Bayes
采用多线程构架实现算法，任务数固定
    设置 prior model 以及 post model for each task，合并所有参数 all parameters = [prior post]
    针对不同任务，利用相应的 post model，计算对应的 empirical loss 与 task complexity
    并结合 meta task complexity items 计算 total objective
    更新所有参数 all parameters，获得更新后的 prior model 与 posterior model
    再更新后的 models 基础上，反复更新迭代，直至获得理想的 prior model
************************************************************************************************************************
"""

from copy import deepcopy

import gym
import torch
import json
import os
import yaml
from tqdm import trange
from functools import reduce
from operator import mul

import maml_rl.envs
from maml_rl.utils.torch_utils import to_numpy
from maml_rl.metalearners.maml_trpo_multi_test1 import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers.multi_task_sampler_multi_test1 import MultiTaskSampler
from maml_rl.utils.helpers_bayes import get_policy_for_env, get_input_size


def main(args, prior_policy=None, init_from_prior=True):
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

    """
    ************************************************************
    新增加的参数：用于获取环境的动作观测空间大小，一便生成随机贝叶斯网络
    output_size = reduce(mul, env.action_space.shape, 1)
    input_size = reduce(mul, env.observation_space.shape, 1)
    ************************************************************
    """
    args.output_size = reduce(mul, env.action_space.shape, 1)
    args.input_size = reduce(mul, env.observation_space.shape, 1)

    # # Policy
    # policy = get_policy_for_env(env,
    #                             hidden_sizes=config['hidden-sizes'],
    #                             nonlinearity=config['nonlinearity'])
    """
    ************************************************************
    新增加的模型：随机网络
    device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')
    log_var_init = {'mean': -10, 'std': 0.1}
    ************************************************************
    """
    # device = args.device
    # log_var_init = args.log_var_init
    # get model:
    # 如果有先验模型，则直接导入先验模型
    if prior_policy and init_from_prior:
        # init from prior model:
        # deepcopy函数：复制并作为一个单独的个体存在；copy函数：复制原有对象，随着原有对象改变而改变
        prior_policy = deepcopy(prior_policy).to(args.device)
    else:
        # 否则直接加载新模型
        prior_policy = get_policy_for_env(args.device,
                                         args.log_var_init,
                                         env,
                                         hidden_sizes=config['hidden-sizes'])

    # 数据无需拷贝，即可使用
    prior_policy.share_memory()

    """
    ************************************************************
    策略 prior model 与 post model 以及对应的参数 param
        prior_policy  posteriors_policies
        prior_params  all_post_param
        all_params
    ************************************************************
    """
    num_tasks = config['meta-batch-size']
    posteriors_policies = [prior_policy for _ in range(num_tasks)]
    all_post_param = sum([list(posterior_policy.parameters()) for posterior_policy in posteriors_policies], [])

    # Create optimizer for all parameters (posteriors + prior)
    # 对所有参数 包括 prior 以及 posterior 创建优化器
    prior_params = list(prior_policy.parameters())
    all_params = all_post_param + prior_params

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    #
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=prior_policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)
    #
    metalearner = MAMLTRPO(prior_policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    # num_iterations = 0

    Info_eposides = {}
    Info_train_loss = []
    Info_valid_loss = []
    Info_tasks = []

    # meta-batch-size：Number of tasks in each batch of tasks
    # 一个batch中任务的个数，此处使用 PAC-Bayes方法，因此任务类型以及数量是固定
    # 也即在2D导航任务中，目标值固定，每次采用不同轨迹进行训练
    tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])

    # 训练的次数 num-batches 个 batch
    for batch in trange(config['num-batches']):

        params_show_train = prior_policy.state_dict()

        # tasks['goal'] fast-batch-size = 20 个目标值 0-19
        # 提取batch中的任务： tasks 为 goal，2D任务中的目标值

        # futures = (train_episodes_futures, valid_episodes_futures)
        # 开始采样轨迹
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        # num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        # num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)

        Info_train_loss.append(logs['train_loss'])
        Info_valid_loss.append(logs['valid_loss'])
        Info_tasks.append(tasks)
        Info_eposides.update(train_episodes=train_episodes,
                             valid_episodes=valid_episodes)

        # logs.update(tasks=tasks,
        #             num_iterations=num_iterations)

        # Save policy
        # 保存网络参数
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                # 保存网络中的参数，f 为路径
                torch.save(prior_policy.state_dict(), f)


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
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    """
    ***************************************************************************
    额外新添加的参数：
    log_var_init = {'mean': -10, 'std': 0.1}  # The initial value for the log-var parameter (rho) of each weight
    n_MC = 1
    lr = 1e-3
    init_from_prior = True
    test_type = 'MaxPosterior'  # 'MaxPosterior' / 'MajorityVote'
    **************************************************************************
    """
    args.log_var_init = {'mean': -10, 'std': 0.1}  # The initial value for the log-var parameter (rho) of each weight
    # Number of Monte-Carlo iterations (for re-parametrization trick):
    args.n_MC = 1
    args.lr = 1e-3
    # No decay
    # args.lr_schedule = {}
    # 是否从利用先验模型进行初始化
    args.init_from_prior = True
    # Test type:
    args.test_type = 'MaxPosterior'  # 'MaxPosterior' / 'MajorityVote'


    main(args)
