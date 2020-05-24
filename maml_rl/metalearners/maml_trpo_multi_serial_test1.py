import torch
from torch import optim

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from Utils.common import grad_step
from maml_rl.samplers.multi_task_sampler_multi_serial_test1 import MultiTaskSampler
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning_StochNN import reinforce_loss


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].

    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """

    # 此处的策略是最原始的策略，采样之前的策略
    def __init__(self,
                 original_policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu'):
        super(MAMLTRPO, self).__init__(original_policy, device=device)
        self.fast_lr = fast_lr
        self.original_policy = original_policy
        self.first_order = first_order

    # 协程函数 async adapt()
    async def adapt(self, train_futures, first_order=None):
        # if first_order is None:
        #     first_order = self.first_order
        # Loop over the number of steps of adaptation 循环调整的步数
        params = None
        # await后面调用future对象，中断当前程序直到得到 futures 的返回值
        # 等待 futures 计算完成，再进行计算 reinforce_loss

        params_show_maml_trpo = self.policy.state_dict()

        for train_future in train_futures:
            """

            """
            train_loss = reinforce_loss(self.original_policy,
                                        await train_future)
            lr = 1e-3
            self.original_policy.train()
            optimizer = optim.Adam(self.original_policy.parameters(), lr)
            # Take gradient step:
            # 计算梯度 已经
            grad_step(train_loss, optimizer)

            """
            原来的算法
            """
            # inner_loss = reinforce_loss(self.policy,
            #                             await futures)
            # # 计算更新后参数，好像不传输到网络中？  self.policy.state_dict()仍然为参数
            # params = self.policy.update_params(inner_loss,
            #                                    params=params,
            #                                    step_size=self.fast_lr,
            #                                    first_order=first_order)

            # params_show_maml_trpo_test = self.policy.state_dict()

        return train_loss

    """
    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    """

    # 协程函数
    # 根据 step() 函数单独计算 task 的 (train, valid) 对应的
    async def surrogate_loss(self,
                             train_futures,
                             valid_futures,
                             old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        # 暂停协程函数，等待协程函数 adapt() 运行结束并更新 self.original_policy
        # 要先在此处暂停函数
        train_loss = await self.adapt(train_futures,
                                      first_order=first_order)

        # valid_loss = reinforce_loss(self.original_policy,
        #                           await futures)

        valid_episodes = await valid_futures
        """

        """
        valid_loss = reinforce_loss(self.original_policy,
                                    valid_episodes)
        """
        要等到上面的 train_futures 进行完之后，再往下进行
        每一个 train_futures 要对应一个 valid_futures，每一对是并行分开运行的
        future 的数量就是 每一个 batch 中 tasks 的数量
        """
        # with torch.set_grad_enabled(old_pi is None):
        #     # 暂停协程函数，等待协程对象 valid_futures 运行结束并输出返回值
        #     valid_episodes = await valid_futures
        #     pi = self.policy(valid_episodes.observations, params=params)
        #
        #     if old_pi is None:
        #         old_pi = detach_distribution(pi)
        #
        #     log_ratio = (pi.log_prob(valid_episodes.actions)
        #                  - old_pi.log_prob(valid_episodes.actions))
        #     ratio = torch.exp(log_ratio)
        #
        #     losses = -weighted_mean(ratio * valid_episodes.advantages,
        #                             lengths=valid_episodes.lengths)
        #     kls = weighted_mean(kl_divergence(pi, old_pi),
        #                         lengths=valid_episodes.lengths)

        return train_loss, valid_loss

    def step(self,
             # futures=(train_episodes_futures, valid_episodes_futures)
             train_futures,
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        # 计算任务数量
        num_tasks = len(train_futures[0])
        logs = {}

        """
        Compute the surrogate loss
        针对每一个 task, 计算 train 和 valid 对，对应的参数之类的
        此处的policy 可以是 GradientBasedMetaLearner 中继承至MultiTaskSampler的采样前的policy
        应该也可以设置为 MAMLTRPO 中的 original_policy，该policy直接来自原始的传参
        有待进一步验证两者 policy 是否一致
        """
        # 此处语句作用是 按 task 依次计算 surrogate_loss() 损失
        train_loss, valid_loss = self._async_gather([
            self.surrogate_loss(train,
                                valid,
                                old_pi=None)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        logs['train_loss'] = to_numpy(train_loss)
        logs['valid_loss'] = to_numpy(valid_loss)

        """
        # 计算平均误差，输出为标量
        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss,
                                    self.policy.parameters(),
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        # 计算平均误差，输出为标量
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # vector_to_parameter( * , self.policy.parameters()) 就是对网络参数的更新
        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

        # 查看最终神经网络参数
        params_final = self.policy.parameters()

        """
        # logs['loss_before', 'kl_before', 'loss_after', 'kl_after']
        return logs
