"""
************************************************************************************************************************
          multi-task Meta RL based on PAC-Bayes

run_test function:
    For a task;
    Based on the post_policy (same as prior_policy), for a task, train post_policy;
    then utilizing trained post_policy, test the effectiveness of post_policy

************************************************************************************************************************
"""

import matplotlib.pyplot as plt

from Utils.common import grad_step
from Utils.complexity_terms import get_task_complexity, get_meta_complexity_term, get_hyper_divergnce
from maml_rl.samplers.multi_task_sampler_multi_serial_test1 import SampleTest

"""
************************************************************************************************************************
run_test function parameters:
    args.optim_func,   args.optim_args,   args.lr_schedule
    task,  prior_policy,  post_policy,  
    num_test_batches,  
************************************************************************************************************************
"""

def run_test(task,
             prior_policy,
             post_policy,
             baseline,
             args,
             env_name,
             env_kwargs,
             batch_size,
             observation_space,
             action_space,
             n_train_tasks,
             num_test_batches=10
             ):
    optim_func, optim_args, lr_schedule =\
        args.optim_func, args.optim_args, args.lr_schedule
    #  Get optimizer:
    optimizer = optim_func(post_policy.parameters(), **optim_args)

    # *******************************************************************
    # Train: post_policy
    # *******************************************************************
    for batch in range(num_test_batches):
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
                                                     n_train_tasks=n_train_tasks)

        sampler = SampleTest(env_name,
                             env_kwargs,
                             batch_size=batch_size,
                             observation_space=observation_space,
                             action_space=action_space,
                             policy=post_policy,
                             baseline=baseline,
                             seed=args.seed,
                             prior_policy=prior_policy,
                             task=task)
        # calculate empirical error for per task
        loss_per_task, avg_reward, last_reward, train_episodes = sampler.sample()

        complexity = get_task_complexity(delta=args.delta,
                                         complexity_type=args.complexity_type,
                                         device=args.device,
                                         divergence_type=args.divergence_type,
                                         kappa_post=args.kappa_post,
                                         prior_model=prior_policy,
                                         post_model=post_policy,
                                         n_samples=batch_size,
                                         avg_empiric_loss=loss_per_task,
                                         hyper_dvrg=hyper_dvrg,
                                         n_train_tasks=n_train_tasks,
                                         noised_prior=True)

        if args.complexity_type == 'Variational_Bayes':
            # note that avg_empiric_loss_per_task is estimated by an average over batch samples,
            #  but its weight in the objective should be considered by how many samples there are total in the task
            n_train_samples = 1
            total_objective = loss_per_task * (n_train_samples) + complexity
        else:
            # 该项类似于 PAC Bayes
            total_objective = loss_per_task + complexity

        # Take gradient step with the shared prior and all tasks' posteriors:
        grad_step(total_objective, optimizer, lr_schedule, args.lr)


    # *******************************************************************
    # Test: post_policy
    # *******************************************************************

    # test_acc, test_loss = run_eval_max_posterior(post_model, test_loader, prm)


    sampler = SampleTest(env_name,
                         env_kwargs,
                         batch_size=batch_size,
                         observation_space=observation_space,
                         action_space=action_space,
                         policy=post_policy,
                         baseline=baseline,
                         seed=args.seed,
                         task=task)
    # calculate empirical error for per task
    test_loss_per_task, test_avg_reward, test_last_reward, train_episodes = sampler.sample()

    Data_post_Trajectory = train_episodes[0].observations.numpy()
    task = task[0]
    task = task['goal']
    plt.plot(Data_post_Trajectory[:,0,0], Data_post_Trajectory[:,0,1])
    plt.plot(task[0], task[1], 'g^')
    plt.savefig('Trajectories.pdf')
    plt.show()
    return test_loss_per_task, test_avg_reward, test_last_reward


# -------------------------------------------------------------------------------------------

# def run_eval_expected(policy, loader, prm):
#     ''' Estimates the expectation of the loss by monte-carlo averaging'''
#     avg_loss = 0.0
#     avg_avg_reward = 0.0
#     n_MC = n_MC  # number of monte-carlo runs for expected loss estimation
#
#     #  monte-carlo runs
#     for i_MC in range(n_MC):
#         sampler = SampleTest(config['env-name'],
#                              env_kwargs=config['env-kwargs'],
#                              batch_size=batch_size,
#                              observation_space=observation_space,
#                              action_space=action_space,
#                              policy=policy,
#                              baseline=baseline,
#                              seed=args.seed,
#                              task=test_task)
#         # calculate empirical error for per task
#         test_loss_per_task, test_avg_reward, test_last_reward = sampler.sample()
#         avg_loss += test_loss_per_task
#         avg_avg_reward += test_avg_reward
#
#     avg_loss /= n_MC
#     avg_avg_reward /= n_MC
#     return avg_loss, avg_avg_reward


# -------------------------选择测试学习后策略的效果方法--------------------------------------
# def run_eval_Bayes(model, loader, prm, verbose=0):
#     # 选择 Bayes 方法
#     with torch.no_grad():    # no need for backprop in test
#
#         if len(loader) == 0:
#             return 0.0, 0.0
#         if prm.test_type == 'Expected':
#             info = run_eval_expected(model, loader, prm)
#         elif prm.test_type == 'MaxPosterior':
#             info = run_eval_max_posterior(model, loader, prm)
#         elif prm.test_type == 'MajorityVote':
#             info = run_eval_majority_vote(model, loader, prm, n_votes=5)
#         elif prm.test_type == 'AvgVote':
#             info = run_eval_avg_vote(model, loader, prm, n_votes=5)
#         else:
#             raise ValueError('Invalid test_type')
#         if verbose:
#             print('Accuracy: {:.3} ({}/{}), loss: {:.4}'.format(float(info['test_acc']), info['n_correct'],
#                                                                           info['n_samples'], float(info['avg_loss'])))
#     return info['acc'], info['avg_loss']
# -------------------------------------------------------------------------------------------


# def run_eval_max_posterior(policy, loader, prm):
#     ''' Estimates the the loss by using the mean network parameters'''
#     # 使用平均网络参数
#     # model.eval()
#     avg_loss = 0
#     n_correct = 0
#     for batch_data in loader:
#         # 提取batch data
#         inputs, targets = data_gen.get_batch_vars(batch_data, prm)
#         batch_size = inputs.shape[0]
#         old_eps_std = policy.set_eps_std(0.0)   # test with max-posterior
#         policy.set_eps_std(old_eps_std)  # return model to normal behaviour
#
#         sampler = SampleTest(config['env-name'],
#                              env_kwargs=config['env-kwargs'],
#                              batch_size=batch_size,
#                              observation_space=observation_space,
#                              action_space=action_space,
#                              policy=policy,
#                              baseline=baseline,
#                              seed=args.seed,
#                              task=test_task)
#         # calculate empirical error for per task
#         test_loss_per_task, test_avg_reward, test_last_reward = sampler.sample()
#
#         return test_loss_per_task, test_avg_reward, test_last_reward
#
#     avg_loss /= n_samples
#     acc = n_correct / n_samples
#     info = {'acc':acc, 'n_correct':n_correct,
#             'n_samples':n_samples, 'avg_loss':avg_loss}
#     return avg_loss, acc

