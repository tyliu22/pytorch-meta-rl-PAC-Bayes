import math

import torch
import torch.multiprocessing as mp
import asyncio
import threading
import time

from datetime import datetime, timezone
from copy import deepcopy

from torch import nn, optim
from torch.distributions import Independent, Normal

from Utils.common import grad_step
from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning_param1 import reinforce_loss


def _create_consumer(queue, futures, loop=None):
    if loop is None:
        # 定义事件循环
        loop = asyncio.get_event_loop()
    while True:
        # 从队列中不断获取数据
        data = queue.get()
        if data is None:
            break
        index, step, episodes = data
        # 定义将要完成任务
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episodes)

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
                 policy,
                 baseline,
                 env=None,
                 seed=None,
                 num_workers=1):
        # 多重继承类，调用类MultiTaskSampler
        super(MultiTaskSampler, self).__init__(env_name,
                                               env_kwargs,
                                               batch_size,
                                               policy,
                                               seed=seed,
                                               env=env)

        self.num_workers = num_workers
        # 初始化队列 训练队列与测试队列 用于提取多进程数据
        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()
        # self.Original_policy = self.policy
        # temporary_policy = self.Original_policy

        # 构建 num_workers 个 workers；调用 num_workers 次
        self.workers = [SamplerWorker(index,
                                      env_name,
                                      env_kwargs,
                                      batch_size,
                                      self.env.observation_space,
                                      self.env.action_space,
                                      self.policy,
                                      deepcopy(baseline),
                                      self.seed,
                                      self.task_queue,
                                      self.train_episodes_queue,
                                      self.valid_episodes_queue,
                                      policy_lock)
            for index in range(num_workers)]

        for worker in self.workers:
            # 守护进程 主进程代码运行结束，守护进程随即终止
            worker.daemon = True
            """
            启动worker (SamplerWorker(index)) 跳转至类SamplerWorker 中的函数 run(self)，
            触发采样训练轨迹，inner更新网络，采样验证轨迹数据
            """
            worker.start()

        self._waiting_sample = False
        # 创建事件循环以及训练、验证双线程
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    # 解压任务 任务数量是 meta-batch-size 每一个batch的中任务的数量
    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    # tasks 是 meta-batch-size 每一个batch的中任务的数量
    # 调用 _start_consumer_threads() 函数，生成 futures
    # futures = (train_episodes_futures, valid_episodes_futures)
    def sample_async(self, tasks, **kwargs):
        if self._waiting_sample:
            raise RuntimeError('Calling `sample_async` while waiting '
                               'for a pending call to `sample_async` '
                               'to complete. Please call `sample_wait` '
                               'before calling `sample_async` again.')

        # 将任务依次添加进队列中
        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs))

        num_steps = kwargs.get('num_steps', 1)
        # 启动训练以及验证线程 返回值为(train_episodes_futures, valid_episodes_futures)
        futures = self._start_consumer_threads(tasks,
                                               num_steps=num_steps)
        self._waiting_sample = True
        # futures = (train_episodes_futures, valid_episodes_futures)
        return futures

    # 根据(train_futures, valid_futures)
    # 输出采样轨迹samples = (train_episodes, valid_episodes)
    def sample_wait(self, episodes_futures):
        if not self._waiting_sample:
            raise RuntimeError('Calling `sample_wait` without any '
                               'prior call to `sample_async`.')
        # 定义协程函数 _wait()
        async def _wait(train_futures, valid_futures):
            # Gather the train and valid episodes 并发运行任务
            # 定义两个异步任务 根据asyncio.gather指令 先并发执行不同的 train_futures
            # 等train_futures任务执行完成后，继续并发执行 valid_futures 任务
            # 最后返回 训练以及验证数据：(train_episodes, valid_episodes)
            train_episodes = await asyncio.gather(*[asyncio.gather(*futures)
                                                  for futures in train_futures])
            valid_episodes = await asyncio.gather(*valid_futures)
            return (train_episodes, valid_episodes)

        # 将协程函数注册到事件循环，并启动事件循环 async def _wait()
        samples = self._event_loop.run_until_complete(_wait(*episodes_futures))
        self._join_consumer_threads()
        self._waiting_sample = False
        return samples

    # 为协程函数 _wait() 指定任务 futures
    def sample(self, tasks, **kwargs):
        futures = self.sample_async(tasks, **kwargs)
        return self.sample_wait(futures)

    @property
    def train_consumer_thread(self):
        if self._train_consumer_thread is None:
            raise ValueError()
        return self._train_consumer_thread

    @property
    def valid_consumer_thread(self):
        if self._valid_consumer_thread is None:
            raise ValueError()
        return self._valid_consumer_thread

    # 启动 训练以及验证 线程
    def _start_consumer_threads(self, tasks, num_steps=1):
        # 开始训练线程
        # Start train episodes consumer thread
        # tasks 每个 task 训练 num_steps
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks]
                                  for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.train_episodes_queue, train_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='train-consumer')
        self._train_consumer_thread.daemon = True
        self._train_consumer_thread.start()

        # Start valid episodes consumer thread
        # 开始训练线程
        #
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.valid_episodes_queue, valid_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='valid-consumer')
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()

        return (train_episodes_futures, valid_episodes_futures)

    def _join_consumer_threads(self):
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None)
            self.train_consumer_thread.join()

        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()

        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return

        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        self._join_consumer_threads()

        self.closed = True


class SamplerWorker(mp.Process):
    # 每一个 worker 都对应于单独的一个SamplerWorker(）
    # 此处的 policy 都来自于里 worker 的 deepcopy(policy)
    #  因此不会对最上面的 policy 作出改变
    def __init__(self,
                 index,
                 env_name,
                 env_kwargs,
                 batch_size,
                 observation_space,
                 action_space,
                 policy,
                 baseline,
                 seed,
                 task_queue,
                 train_queue,
                 valid_queue,
                 policy_lock):
        super(SamplerWorker, self).__init__()

        # 为 batch 中 batch_size 个任务都创建环境 (num_batches * batch_size)
        env_fns = [make_env(env_name, env_kwargs=env_kwargs)
                   for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_fns,
                                  observation_space=observation_space,
                                  action_space=action_space)
        self.envs.seed(None if (seed is None) else seed + index * batch_size)
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.policy_lock = policy_lock

    def sample(self,
               index,
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
        # 此处参数设置为 None，调用 OrderDict() 参数
        """
        ******************************************************************
        """
        # params = None

        # params_show_multi_task_sampler = self.policy.state_dict()

        for step in range(num_steps):
            # 获取该batch中所有的轨迹数据，将数据保存至 train_episodes
            train_episodes = self.create_episodes(gamma=gamma,
                                                  gae_lambda=gae_lambda,
                                                  device=device)
            train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
            # QKFIX: Deep copy the episodes before sending them to their
            # respective queues, to avoid a race condition. This issue would 
            # cause the policy pi = policy(observations) to be miscomputed for
            # some timesteps, which in turns makes the loss explode.
            self.train_queue.put((index, step, deepcopy(train_episodes)))

            """
                计算 reinforce loss， 更新网络参数 params
            """
            # 多线程程序中，安全使用可变对象
            # with + lock：保证每次只有一个线程执行下面代码块
            # with 语句会在这个代码块执行前自动获取锁，在执行结束后自动释放锁
            with self.policy_lock:
                """
                ******************************************************************
                """
                loss = reinforce_loss(self.policy, train_episodes)
                lr = 1e-3
                self.policy.train()
                optimizer = optim.Adam(self.policy.parameters(), lr)
                # Take gradient step:
                # 计算梯度 已经
                grad_step(loss, optimizer)
                # params = self.policy.update_params(loss,
                #                                    params=params,
                #                                    step_size=fast_lr,
                #                                    first_order=True)
                """
                ******************************************************************
                """
                # params_show_multi_task_sampler_test = self.policy.state_dict()

        # Sample the validation trajectories with the adapted policy
        valid_episodes = self.create_episodes(gamma=gamma,
                                              gae_lambda=gae_lambda,
                                              device=device)
        valid_episodes.log('_enqueueAt', datetime.now(timezone.utc))
        self.valid_queue.put((index, None, deepcopy(valid_episodes)))

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
        episodes.log('process_name', self.name)

        #
        t0 = time.time()
        """
        ******************************************************************
        """
        for item in self.sample_trajectories():
            episodes.append(*item)
        episodes.log('duration', time.time() - t0)

        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        return episodes

    def sample_trajectories(self,
                            init_std=1.0,
                            min_std=1e-6,
                            output_size=2):
        # 基于当前策略，采样 batch_size 个完整的轨迹
        observations = self.envs.reset()
        with torch.no_grad():
            while not self.envs.dones.all():
                observations_tensor = torch.from_numpy(observations)
                """
                ******************************************************************
                """
                output = self.policy(observations_tensor)

                min_log_std = math.log(min_std)
                sigma = nn.Parameter(torch.Tensor(output_size))
                sigma.data.fill_(math.log(init_std))

                scale = torch.exp(torch.clamp(sigma, min=min_log_std))
                # loc 是高斯分布均值
                # scale 是高斯分布方差
                p_normal = Independent(Normal(loc=output, scale=scale), 1)

                actions_tensor = p_normal.sample()
                actions = actions_tensor.cpu().numpy()

                # pi = policy(observations_tensor)
                # actions_tensor = pi.sample()
                # actions = actions_tensor.cpu().numpy()

                new_observations, rewards, _, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']
                yield (observations, actions, rewards, batch_ids)
                observations = new_observations

    # start
    def run(self):
        while True:
            data = self.task_queue.get()

            if data is None:
                self.envs.close()
                self.task_queue.task_done()
                break

            # index task['goal']  kwargs:params
            index, task, kwargs = data
            self.envs.reset_task(task)
            """
            sample
            """
            self.sample(index, **kwargs)
            self.task_queue.task_done()
