import math

import torch
import time

from datetime import datetime, timezone
from copy import deepcopy

from torch import nn, optim
from torch.distributions import Independent, Normal

from Utils.common import grad_step
from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss


# 多线程设计
class MultiTaskSampler(Sampler):
    """Vectorized sampler to sample trajectories from multiple environements.

    Parameters
    ----------
    env_name : str
        Name of the environment. This environment should be an environment
        registered through `gym`. See `maml.envs`.

    env_kwargs : dict
        Additional keywork arguments to be added when creating the environment.

    batch_size : int  每一个任务采样的轨迹数
        Number of trajectories to sample from each task (ie. `fast_batch_size`).

    policy : `maml_rl.policies.Policy` instance
        The policy network for sampling. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    baseline : `maml_rl.baseline.LinearFeatureBaseline` instance
        The baseline. This baseline is an instance of `nn.Module`, with an
        additional `fit` method to fit the parameters of the model.

    env : `gym.Env` instance (optional)
        An instance of the environment given by `env_name`. This is used to
        sample tasks from. If not provided, an instance is created from `env_name`.

    seed : int (optional)
        Random seed for the different environments. Note that each task and each
        environement inside every process use different random seed derived from
        this value if provided.

    num_workers : int   处理核心进程的数量，不同于每一个batch中任务的数量
        Number of processes to launch. Note that the number of processes does
        not have to be equal to the number of tasks in a batch (ie. `meta_batch_size`),
        and can scale with the amount of CPUs available instead.
    """

    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 num_tasks,
                 policy,
                 baseline,
                 env=None,
                 seed=None):

        baseline = deepcopy(baseline)
        # 为 batch 中 batch_size 个任务都创建环境 (num_batches * batch_size)

        # env_fns = [make_env(env_name, env_kwargs=env_kwargs)
        #            for _ in range(batch_size)]
        # self.envs = SyncVectorEnv(env_fns,
        #                           observation_space=env.observation_space,
        #                           action_space=env.action_space)
        # self.envs.seed(None if (seed is None) else seed + index * batch_size)
        self.tasks = self.sample_tasks(env, num_tasks)
        env_fns = [make_env(env_name, env_kwargs=env_kwargs)]
        self.env = SyncVectorEnv(env_fns,
                                 observation_space=env.observation_space,
                                 action_space=env.action_space)
        self.env.seed(None if (seed is None) else seed)

        # self.env = env
        # self.env.seed(None if (seed is None) else seed)
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        # task = self.sample_tasks()
        # self.envs.reset_task(task)

    def sample_tasks(self, env, num_tasks):
        return env.unwrapped.sample_tasks(num_tasks)
    # def sample_tasks(self, num_tasks):
    #     return self.env.unwrapped.sample_tasks(num_tasks)

    def sample_tasks_return(self):
        return self.tasks

    # def sample_batch_size(self):
    #     train_episodes, train_loss, \
    #     valid_episodes, valid_loss = [sample(num_steps=1,
    #                                          fast_lr=0.5,
    #                                          gamma=0.95,
    #                                          gae_lambda=1.0,
    #                                          device='cpu')
    #         for _ in range(self.batch_size)
    #     ]
    #     return train_episodes, train_loss, valid_episodes, valid_loss

    def sample(self,
               task=None,
               num_steps=1,
               fast_lr=0.5,
               gamma=0.95,
               gae_lambda=1.0,
               device='cpu'):
        """
        基于初始策略采样训练轨迹，并基于REINFORCE损失调整策略
        内循环中，梯度更新使用`first_order=True`，因其仅用于采样轨迹，而不是优化
        Sample the training trajectories with the initial policy and adapt the
        policy to the task, based on the REINFORCE loss computed on the
        training trajectories. The gradient update in the fast adaptation uses
        `first_order=True` no matter if the second order version of MAML is
        applied since this is only used for sampling trajectories, and not
        for optimization.
        """
        """
        训练阶段：
            采样训练轨迹数据 train_episodes，计算loss，更新原有网络参数
            采样验证轨迹数据 valid_episodes
        MAML 内部循环更新num_steps次 inner loop / fast adaptation
        """

        self.env.reset_task(task)

        # # 此处参数设置为 None，调用 OrderDict() 参数
        # params = None
        #
        # params_show_multi_task_sampler = self.policy.state_dict()
        # train_episodes = []
        # 先采样训练阶段数据轨迹
        for step in range(num_steps):
            # 获取该batch中所有的轨迹数据，将数据保存至 train_episodes
            # for i in range(self.batch_size):
            train_episodes = self.create_episodes(gamma=gamma,
                                                  gae_lambda=gae_lambda,
                                                  device=device)

            train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
            # QKFIX: Deep copy the episodes before sending them to their
            # respective queues, to avoid a race condition. This issue would 
            # cause the policy pi = policy(observations) to be miscomputed for
            # some timesteps, which in turns makes the loss explode.

            """
                计算 reinforce loss， 更新网络参数 params
            """

            # 多线程程序中，安全使用可变对象
            # with + lock：保证每次只有一个线程执行下面代码块
            # with 语句会在这个代码块执行前自动获取锁，在执行结束后自动释放锁
            train_loss = reinforce_loss(self.policy, train_episodes)
            train_loss = train_loss.mean()
            lr = 1e-3
            self.policy.train()
            optimizer = optim.Adam(self.policy.parameters(), lr)
            # Take gradient step:
            # 计算梯度 已经
            grad_step(train_loss, optimizer)

            # params_show_multi_task_sampler_test = self.policy.state_dict()

        # Sample the validation trajectories with the adapted policy
        valid_episodes = self.create_episodes(gamma=gamma,
                                              gae_lambda=gae_lambda,
                                              device=device)
        valid_loss = reinforce_loss(self.policy, valid_episodes)
        valid_episodes.log('_enqueueAt', datetime.now(timezone.utc))

        return train_episodes, train_loss, valid_episodes, valid_loss

    # 构建 episodes 变量，用于保存完整轨迹的数据
    # episodes = (observation, action, reward, batch_ids, advantage)
    def create_episodes(self,
                        gamma=0.95,
                        gae_lambda=1.0,
                        device='cpu'):
        # 初始化 episodes，用于保存 完整的轨迹数据
        # 将sample_trajectories函数采样 batch_size 个完整的轨迹保存至 episodes
        episodes = BatchEpisodes(batch_size=self.batch_size,
                                 gamma=gamma,
                                 device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))

        #
        t0 = time.time()
        for item in self.sample_trajectories():
            episodes.append(*item)
        episodes.log('duration', time.time() - t0)

        # self.baseline.fit(episodes)
        # episodes.compute_advantages(self.baseline,
        #                             gae_lambda=gae_lambda,
        #                             normalize=True)
        return episodes

    def sample_trajectories(self,
                            init_std=1.0,
                            min_std=1e-6,
                            output_size=2):
        # 基于当前策略，采样 batch_size 个完整的轨迹
        observations = self.env.reset()
        with torch.no_grad():
            while not self.env.dones.all():
                observations_tensor = torch.from_numpy(observations)
                output = self.policy(observations_tensor)

                min_log_std = math.log(min_std)
                sigma = nn.Parameter(torch.Tensor(output_size))
                sigma.data.fill_(math.log(init_std))

                scale = torch.exp(torch.clamp(sigma, min=min_log_std))
                p_normal = Independent(Normal(loc=output, scale=scale), 1)

                actions_tensor = p_normal.sample()
                actions = actions_tensor.cpu().numpy()

                new_observations, rewards, _, infos = self.env.step(actions)
                batch_ids = infos['batch_ids']
                yield observations, actions, rewards, batch_ids
                observations = new_observations
