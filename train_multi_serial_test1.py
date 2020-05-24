"""
************************************************************************************************************************
          multi-task Meta RL based on PAC-Bayes

针对 Multi

当前

采用串行 serial 构架架实现算法，任务数固定
    设置 prior model 以及 post model for each task，合并所有参数 all parameters = [prior post]
    针对不同任务，利用相应的 post model，计算对应的 empirical loss 与 task complexity
    并结合 meta task complexity items 计算 total objective
    更新所有参数 all parameters，获得更新后的 prior model 与 posterior model
    再更新后的 models 基础上，反复更新迭代，直至获得理想的 prior model
************************************************************************************************************************
# """

from copy import deepcopy

import gym
import torch
import json
import os
import yaml
from tqdm import trange
from functools import reduce
from operator import mul
import torch.optim as optim

import maml_rl.envs
from maml_rl.utils.torch_utils import to_numpy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers.multi_task_sampler_multi_serial_test1 import SampleTest
from maml_rl.utils.helpers_bayes import get_policy_for_env, get_input_size
from Utils.complexity_terms import get_task_complexity, get_meta_complexity_term, get_hyper_divergnce
from Utils.common import grad_step
from Utils.PlotTrajectories import plotTrajectories
import run_test


def main(args, prior_policy=None, init_from_prior=True):

    # *******************************************************************
    # config log filename
    #    'r': read;  'w': write
    # *******************************************************************
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        # 如果没有文件，则创建文件地址
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        # 文件夹地址与文件名
        policy_filename = os.path.join(args.output_folder, 'policy_2d_PAC_Bayes.th')
        config_filename = os.path.join(args.output_folder, 'config_2d_PAC_Bayes.json')

        # with open(config_filename, 'w') as f:
        #     config.update(vars(args))
        #     json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    # 待测试
    env.seed(args.seed)
    env.close()

    """
    ************************************************************
    新增加的参数：用于获取环境的动作观测空间大小，一便生成随机贝叶斯网络
    output_size = reduce(mul, env.action_space.shape, 1)
    input_size = reduce(mul, env.observation_space.shape, 1)
    ************************************************************
    """
    observation_space = env.observation_space
    action_space = env.action_space
    args.output_size = reduce(mul, env.action_space.shape, 1)
    args.input_size = reduce(mul, env.observation_space.shape, 1)

    """
    ************************************************************
    新增加的模型：随机网络
    device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')
    log_var_init = {'mean': -10, 'std': 0.1}
    ************************************************************
    """
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
    # prior_policy.share_memory()

    """
    ************************************************************
    策略 prior model 与 post model 以及对应的参数 param
        prior_policy  posteriors_policies
        prior_params  all_post_param
        all_params
    ************************************************************
    """
    num_tasks = config['meta-batch-size']
    batch_size = config['fast-batch-size']

    # Unpack parameters:
    # 提取参数 优化方法 优化参数 学习率等
    optim_func, optim_args, lr_schedule =\
        args.optim_func, args.optim_args, args.lr_schedule

    posteriors_policies = [get_policy_for_env(args.device,
                           args.log_var_init,
                           env,
                           hidden_sizes=config['hidden-sizes']) for _ in range(num_tasks)]
    all_post_param = sum([list(posterior_policy.parameters()) for posterior_policy in posteriors_policies], [])

    # Create optimizer for all parameters (posteriors + prior)
    # 对所有参数 包括 prior 以及 posterior 创建优化器
    prior_params = list(prior_policy.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)

    """生成固定的 tasks
        随机数问题尚未解决，可重复性不行
    """
    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # 生成 'meta-batch-size' 任务
    # for task in enumerate(tasks):
    tasks = env.unwrapped.sample_tasks(num_tasks)

    # meta-batch-size：Number of tasks in each batch of tasks
    # 一个batch中任务的个数，此处使用 PAC-Bayes方法，因此任务类型以及数量是固定
    # 也即在2D导航任务中，目标值固定，每次采用不同轨迹进行训练
    # tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])

    avg_empiric_loss_per_task = torch.zeros(num_tasks, device=args.device)
    avg_reward_per_task = torch.zeros(num_tasks, device=args.device)
    complexity_per_task = torch.zeros(num_tasks, device=args.device)
    # 此参数针对不同任务有不同的训练数量的情况
    n_samples_per_task = torch.zeros(num_tasks, device=args.device)

    Info_avg_reward = []
    Info_total_objective = []
    Info_last_reward = []
    Info_train_trajectories = []


    # 训练的次数 num-batches 个 batch
    for batch in range(config['num-batches']):
        # print(batch)

        # params_show_train = prior_policy.state_dict()

        # Hyper-prior term:
        # 计算超先验与超后验的散度
        hyper_dvrg = get_hyper_divergnce(kappa_prior=args.kappa_prior,
                                         kappa_post=args.kappa_post,
                                         divergence_type=args.divergence_type,
                                         device=args.device,
                                         prior_model=prior_policy)
        # 根据 超散度 hyper_dvrg 计算对应的 meta项  传参方式也可以直接安顺序传递
        meta_complex_term = get_meta_complexity_term(hyper_kl=hyper_dvrg,
                                                     delta=args.delta,
                                                     complexity_type=args.complexity_type,
                                                     n_train_tasks=num_tasks)

        for i_task in range(num_tasks):
            sampler = SampleTest(config['env-name'],
                                 env_kwargs=config['env-kwargs'],
                                 batch_size=batch_size,
                                 observation_space=observation_space,
                                 action_space=action_space,
                                 policy=posteriors_policies[i_task],
                                 baseline=baseline,
                                 seed=args.seed,
                                 prior_policy=prior_policy,
                                 task=tasks[i_task])
            # calculate empirical error for per task
            loss_per_task, avg_reward, last_reward, train_episodes = sampler.sample()

            complexity = get_task_complexity(delta=args.delta,
                                             complexity_type=args.complexity_type,
                                             device=args.device,
                                             divergence_type=args.divergence_type,
                                             kappa_post=args.kappa_post,
                                             prior_model=prior_policy,
                                             post_model=posteriors_policies[i_task],
                                             n_samples=batch_size,
                                             avg_empiric_loss=loss_per_task,
                                             hyper_dvrg=hyper_dvrg,
                                             n_train_tasks=num_tasks,
                                             noised_prior=True)

            avg_empiric_loss_per_task[i_task] = loss_per_task
            avg_reward_per_task[i_task] = avg_reward
            complexity_per_task[i_task] = complexity
            n_samples_per_task[i_task] = batch_size

        # Approximated total objective:
        if args.complexity_type == 'Variational_Bayes':
            # note that avg_empiric_loss_per_task is estimated by an average over batch samples,
            #  but its weight in the objective should be considered by how many samples there are total in the task
            total_objective = \
                (avg_empiric_loss_per_task * n_samples_per_task + complexity_per_task).mean() * num_tasks \
                + meta_complex_term
            # total_objective = ( avg_empiric_loss_per_task * n_samples_per_task
            # + complexity_per_task).mean() + meta_complex_term

        else:
            total_objective = \
                avg_empiric_loss_per_task.mean() + complexity_per_task.mean() + meta_complex_term

        # Take gradient step with the shared prior and all tasks' posteriors:
        grad_step(total_objective, all_optimizer, lr_schedule, args.lr)

        Info_avg_reward.append(avg_reward_per_task.mean())
        Info_total_objective.append(total_objective)
        Info_last_reward.append(last_reward)

    # *******************************************************************
    # Save policy
    # *******************************************************************
    # 将模型参数保存至 policy_filename 中的 python.th
    if args.output_folder is not None:
        with open(policy_filename, 'wb') as f:
            # 保存网络中的参数，f 为路径
            torch.save(prior_policy.state_dict(), f)


    # *******************************************************************
    # Test
    # learned policy   : prior_policy
    # saved parameters : 'policy_2d_PAC_Bayes.th'
    # *******************************************************************
    env_name = config['env-name'],
    env_kwargs = config['env-kwargs']
    test_num = 10

    Info_test_loss = []
    Info_test_avg_reward = []
    Info_test_last_reward = []

    for test_batch in range(test_num):
        # 生成新任务，训练并进行验证误差
        test_task = env.unwrapped.sample_tasks(1)
        post_policy = get_policy_for_env(args.device,
                                         args.log_var_init,
                                         env,
                                         hidden_sizes=config['hidden-sizes'])
        post_policy.load_state_dict(prior_policy.state_dict())

        # based on the prior_policy, train post_policy; then test learned post_policy
        test_loss_per_task, test_avg_reward, test_last_reward = run_test(task=test_task,
                                                                         prior_policy=prior_policy,
                                                                         post_policy=post_policy,
                                                                         baseline=baseline,
                                                                         args=args,
                                                                         env_name=env_name,
                                                                         env_kwargs=env_kwargs,
                                                                         batch_size=batch_size,
                                                                         observation_space=observation_space,
                                                                         action_space=action_space,
                                                                         n_train_tasks=num_tasks)

        Info_test_loss.append(test_loss_per_task)
        Info_test_avg_reward.append(test_avg_reward)
        Info_test_last_reward.append(test_last_reward)


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
    kappa_prior = 1e2
    kappa_post = 1e-3
    divergence_type = 'KL'
    **************************************************************************
    """

    """随机网络参数设置：初始化参数等
    """
    # The initial value for the log-var parameter (rho) of each weight
    args.log_var_init = {'mean': -10, 'std': 0.1}
    # Number of Monte-Carlo iterations (for re-parametrization trick):
    args.n_MC = 1

    # 是否从利用先验模型进行初始化
    args.init_from_prior = True
    # Test type:
    args.test_type = 'MaxPosterior'  # 'MaxPosterior' / 'MajorityVote'

    """超先验 / 超后验 参数设置
    """
    # MPB alg  params:
    #  parameter of the hyper-prior regularization
    args.kappa_prior = 1e2
    # The STD of the 'noise' added to prior
    args.kappa_post = 1e-3

    """计算两分布距离的方法：KL：KL散度；
    """
    # 'KL' or 'W_NoSqr' or 'W_Sqr'
    args.divergence_type = 'KL'

    # maximal probability that the bound does not hold
    args.delta = 0.1
    """不同的PAC-Bayes理论：McAllester；Seeger；Pentina；VB；Catoni；NoComplexity
    """
    # " The learning objective complexity type"
    # 'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina' / 'McAllester' / 'Seeger' / 'Catoni'
    args.complexity_type = 'McAllester'

    """优化器参数设置：optim_func；optim_args；lr_schedule
        """
    args.lr = 5e-3
    #  Define optimizer:
    args.optim_func, args.optim_args = optim.Adam, {'lr': args.lr}  # 'weight_decay': 1e-4
    # prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr, 'amsgrad': True}  #'weight_decay': 1e-4
    # prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

    # Learning rate decay schedule:
    # prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
    args.lr_schedule = {}  # No decay

    main(args)
