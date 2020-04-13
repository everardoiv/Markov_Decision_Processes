from hiive import mdptoolbox
from hiive.mdptoolbox import example, mdp

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import gym.wrappers as MyWrapper
from qlearner import QLearner
from value_iteration import value_iteration
from policy_iteration import policy_iteration

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def plot_stats(stats, name, y_col='Error'):
    df = pd.DataFrame.from_records(stats)
    print(df.tail(5))
    plt.clf()
    plt.title(('%s, time - %0.3f' % (name, df['Time'].max())))
    plt.xlabel('Iterations')
    plt.ylabel(y_col)
    df.plot(x='Iteration', y=y_col, kind='line')
    plt.tight_layout()
    plt.savefig('plots/frozen_lakes/single_episode/%s_%s.png' % (name, y_col))

    plt.clf()
    plt.title(('%s, time - %0.3f' % (name, df['Time'].sum())))
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    df.plot(x='Iteration', y=y_col, kind='line')
    plt.tight_layout()
    plt.savefig('plots/frozen_lakes/single_episode/reward_v_iteration_%s.png' % name)

    plt.clf()
    plt.title('Iterations v Time')
    plt.xlabel('Time')
    plt.ylabel('Iterations')
    df.plot(x='Time', y='Iteration', kind='line')
    plt.tight_layout()
    plt.savefig('plots/frozen_lakes/single_episode/iterations_v_time_%s.png' % name)


def timing(func):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = func(*args, **kwargs)
        time2 = time.time()
        print("%s function took %0.3f ms" % (func.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


@timing
def mdp_example():
    # S are the states of the forest with S - 1 being the oldest
    P, R = mdptoolbox.example.forest()
    print("== Forest Example ==")
    print("Transition Array: ")
    print(P.shape)
    print(P)  # Transition array A x S x S
    print("Reward Array: ")
    print(R.shape)
    print(R)  # Reward array S x A
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi.run()
    print("Optimal Policy: ")
    print(vi.policy)
    print()


@timing
def run_gridworld(env_name, mapping, shape=None, new_lake=None):
    np.random.seed(42)
    min_r = -100.0
    max_r = 101.0
    env = MyWrapper.TransformReward(gym.make(env_name, desc=new_lake), lambda r: np.clip(r*100.0, min_r, max_r))
    env.reset()
    print("== %s ==" % env_name)
    print("Actions: " + str(env.env.action_space.n))
    print("States: " + str(env.env.observation_space.n))
    print(np.asarray(env.env.desc, dtype="U"))
    print()

    P, R = evaluate_rewards_and_transitions(env, False)

    gamma_range = np.append(np.linspace(0.1, 0.9, 9), np.linspace(0.91, 0.99, 9))

    frozen_lake_all(P, R, gamma_range, mapping, shape)

    frozen_lake_vi(P, R, gamma_range, mapping, shape)

    frozen_lake_pi(P, R, gamma_range, mapping, shape)

    frozen_lake_pim(P, R, gamma_range, mapping, shape)

    frozen_lake_ql(P, R, gamma_range, mapping, shape)

    comparing_mdps(P, R, mapping, shape)


def frozen_lake_all(P, R, gamma_range, mapping, shape):

    vi_iteration_list = np.zeros(gamma_range.shape)
    vi_time_list = np.zeros(gamma_range.shape)
    vi_reward_list = np.zeros(gamma_range.shape)
    vi_error_list = np.zeros(gamma_range.shape)

    pi_iteration_list = np.zeros(gamma_range.shape)
    pi_time_list = np.zeros(gamma_range.shape)
    pi_reward_list = np.zeros(gamma_range.shape)
    pi_error_list = np.zeros(gamma_range.shape)

    diff_list = np.zeros(gamma_range.shape)

    expected_policy = None

    for i, gamma in enumerate(gamma_range):
        print('Gamma %0.2f' % gamma)

        vi = mdp.ValueIteration(transitions=P, reward=R, gamma=gamma, epsilon=0.0001, max_iter=5000)
        # vi.setVerbose()
        vi.run()

        vi_iteration_list[i] = vi.run_stats[-1:][0]['Iteration']
        vi_time_list[i] = vi.run_stats[-1:][0]['Time']
        vi_reward_list[i] = vi.run_stats[-1:][0]['Reward']
        vi_error_list[i] = vi.run_stats[-1:][0]['Error']

        pi = mdp.PolicyIteration(transitions=P, reward=R, gamma=gamma, max_iter=5000, eval_type=1)
        # pi.setVerbose()
        pi.run()

        pi_iteration_list[i] = pi.run_stats[-1:][0]['Iteration']
        pi_time_list[i] = pi.run_stats[-1:][0]['Time']
        pi_reward_list[i] = pi.run_stats[-1:][0]['Reward']
        pi_error_list[i] = pi.run_stats[-1:][0]['Error']

        print('Value Iteration Policy Found: ' + str(vi.policy))
        print_policy(vi.policy, mapping, shape)
        print('Policy Iteration Policy Found: ' + str(pi.policy))
        print_policy(pi.policy, mapping, shape)

        difference1 = sum([abs(x - y) for x, y in zip(pi.policy, vi.policy)])
        diff_list[i] = difference1
        print('Discrepancy in Policy and Value Iteration: ', difference1)

        if difference1 == 0:
            expected_policy = vi.policy

        print()

        # Plotting
        # Error v Iteration
        plt.clf()
        plt.title('Value Iteration: Error v Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.plot(list(vi_iteration_list), list(vi_error_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/vi_error_v_iteration.png')

        # Reward v Gamma
        plt.clf()
        plt.title('Value Iteration: Reward v Gamma')
        plt.xlabel('Gamma')
        plt.ylabel('Reward')
        plt.plot(list(gamma_range), list(vi_reward_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/vi_reward_v_gamma.png')

        # Gamma v Iterations
        plt.clf()
        plt.title('Value Iteration: Gamma v Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Gamma')
        plt.plot(list(vi_iteration_list), list(gamma_range))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/vi_gamma_v_iterations.png')

        # Gamma v Time
        plt.clf()
        plt.title('Value Iteration: Gamma v Time')
        plt.xlabel('Time')
        plt.ylabel('Gamma')
        plt.plot(list(vi_time_list), list(gamma_range))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/vi_gamma_v_time.png')

        # Reward vs Iterations
        plt.clf()
        plt.title('Value Iteration: Reward v Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.plot(list(vi_iteration_list), list(vi_reward_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/vi_reward_v_iterations.png')

        # Policy
        # Error v Iteration
        plt.clf()
        plt.title('Policy Iteration: Error v Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.scatter(list(pi_iteration_list), list(pi_error_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/pi_error_v_iteration.png')

        # Gamma v Reward
        plt.clf()
        plt.title('Policy Iteration: Reward v Gamma')
        plt.xlabel('Gamma')
        plt.ylabel('Reward')
        plt.scatter(list(gamma_range), list(pi_reward_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/pi_reward_v_gamma.png')

        # Gamma v Iterations
        plt.clf()
        plt.title('Policy Iteration: Gamma v Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Gamma')
        plt.scatter(list(pi_iteration_list), list(gamma_range))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/pi_gamma_v_iterations.png')

        # Gamma v Time
        plt.clf()
        plt.title('Policy Iteration: Gamma v Time')
        plt.xlabel('Time')
        plt.ylabel('Gamma')
        plt.scatter(list(pi_time_list), list(gamma_range))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/pi_gamma_v_time.png')

        # Reward vs Iterations
        plt.clf()
        plt.title('Policy Iteration: Reward v Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.scatter(list(pi_iteration_list), list(pi_reward_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/pi_reward_v_iterations.png')

        # Gamma vs Policy Differences
        plt.clf()
        plt.title('Gamma v Policy Differences')
        plt.xlabel('Gamma')
        plt.ylabel('Policy Differences')
        plt.scatter(list(gamma_range), list(diff_list))
        plt.tight_layout()
        plt.savefig('plots/frozen_lakes/gamma_v_differences.png')

    # TODO
    gamma_range = np.array([0.8, 0.9, 0.99])
    alpha_range = np.array([0.1, 0.9, 0.99])
    epsilon_range = np.array([0.1, 0.5, 0.9, 0.999])
    e_decay_range = np.array([0.1, 0.5, 0.9, 0.999])

    # alpha_range = np.append(np.linspace(0.01, 0.1, 9), np.linspace(0.2, 0.99, 4))
    # epsilon_range = np.linspace(0.1, 1.0, 10)
    # e_decay_range = np.append(np.linspace(0.1, 0.9, 4), np.linspace(0.91, 0.99, 9))

    prev_Q = None
    thresh = 1e-4
    print('== Q Learning ==')
    for i, gamma in enumerate(gamma_range):
        for j, alpha in enumerate(alpha_range):
            for k, ep in enumerate(epsilon_range):
                for l, ed in enumerate(e_decay_range):
                    # print('ql: gamma - {}, alpha - {}, epsilon - {}, e_decay - {}'.format(gamma, alpha, ep, ed))
                    ql = mdp.QLearning(transitions=P, reward=R, gamma=gamma, alpha=alpha, alpha_decay=1.0,
                                       alpha_min=0.001,
                                       epsilon=ep, epsilon_min=0.1, epsilon_decay=ed, n_iter=10e4)
                    stats = ql.run()
                    plot_stats(stats, ('ql_frozen_lake_%0.2f_%0.2f_%0.2f_%0.2f' % (gamma, alpha, ep, ed)))

                    # print('Policy: ')
                    # print(ql.policy)
                    # print(ql.run_stats)
                    df = pd.DataFrame.from_records(ql.run_stats)
                    iteration_list = df['Iteration'][-100:]
                    windowed_reward = df['Reward'][-100:].mean()
                    error_list = df['Error'][-100:].mean()

                    if prev_Q is None:
                        prev_Q = ql.Q
                    else:
                        variation = np.absolute(np.subtract(np.asarray(ql.Q), np.asarray(prev_Q))).max()
                        res = np.abs(np.subtract(np.asarray(prev_Q), np.asarray(ql.Q)))
                        print('Result: ')
                        print(res)
                        print('Variation: ')
                        print(variation)
                        print('Mean Reward for Last 100 Iterations:')
                        print(windowed_reward)
                        if np.all(res < thresh) or variation < thresh or windowed_reward > 45.0:
                            print('Breaking! Below Thresh')
                            print('Found at: gamma - {}, alpha - {}, epsilon - {}, e_decay - {}'.format(
                                gamma, alpha, ep, ed))
                            print('Optimal Policy: ')
                            print(ql.policy)
                            break

                # Epsilon Decay vs. Iterations

            # Epsilon vs. Iterations

    # Plotting


    # Error v Iteration

    # Gamma v Reward

    # Gamma v Iterations

    # Gamma v Time

    # Iteration v Reward


def comparing_mdps(P, R, mapping, shape):
    print("Comparing the Two Policies")
    vi = mdp.ValueIteration(P, R, 0.9, max_iter=10000)
    vi.run()
    print("Value Function: ")
    print(vi.V)
    print("Policy: ")
    print(vi.policy)
    print_policy(vi.policy, mapping, shape)
    print("Iter: ")
    print(vi.iter)
    print("Time: ")
    print(vi.time)
    # print(vi.run_stats)
    print()
    pi = mdp.PolicyIteration(P, R, 0.9, max_iter=100000)
    pi.run()
    print("Policy Function: ")
    print(pi.V)
    print("Policy: ")
    print(pi.policy)
    print_policy(pi.policy, mapping, shape)
    print("Iter: ")
    print(pi.iter)
    print("Time: ")
    print(pi.time)
    # print(pi.run_stats)
    print()
    pim = mdp.PolicyIterationModified(P, R, 0.9, max_iter=100000, epsilon=0.05)
    pim.run()
    print("Policy Modified Function: ")
    print(pim.V)
    print("Policy: ")
    print(pim.policy)
    print_policy(pim.policy, mapping, shape)
    print("Iter: ")
    print(pim.iter)
    print("Time: ")
    print(pim.time)
    # print(pi.run_stats)
    print()
    ql = mdp.QLearning(
        P, R, 0.9, n_iter=10e4, epsilon=0.1, epsilon_decay=0.1, epsilon_min=0.1,
    )
    ql.run()
    print("Q Learning Function: ")
    print(ql.V)
    print("Policy: ")
    print(ql.policy)
    print_policy(ql.policy, mapping, shape)
    print("Mean Discrepancy: ")
    print(ql.error_mean)
    # print(ql.v_mean)
    print("Epsilon: ")
    print(ql.epsilon)
    difference1 = sum([abs(x - y) for x, y in zip(pi.policy, vi.policy)])
    if difference1 > 0:
        print("Discrepancy in Policy and Value Iteration: ", difference1)
        print()
    difference2 = sum([abs(x - y) for x, y in zip(pim.policy, vi.policy)])
    if difference2 > 0:
        print("Discrepancy in Policy Modified and Value Iteration: ", difference2)
        print()
    difference3 = sum([abs(x - y) for x, y in zip(pim.policy, pi.policy)])
    if difference3 > 0:
        print("Discrepancy in Policy Modified and Policy Iteration: ", difference3)
        print()
    difference4 = sum([abs(x - y) for x, y in zip(vi.policy, ql.policy)])
    if difference4 > 0:
        print("Discrepancy in Q Learning and Value Iteration: ", difference4)
        print()
    difference5 = sum([abs(x - y) for x, y in zip(pi.policy, ql.policy)])
    if difference5 > 0:
        print("Discrepancy in Q Learning and Policy Iteration: ", difference5)
        print()
    difference6 = sum([abs(x - y) for x, y in zip(pim.policy, ql.policy)])
    if difference6 > 0:
        print("Discrepancy in Q Learning and Policy Iteration Modified: ", difference6)
        print()


def frozen_lake_ql(P, R, gamma_range, mapping, shape):
    print("== Q Learning Iteration ==")
    print("gamma    #Iterations     time (ms)")
    prev_policy = []
    prev_gamma = 0
    no_diff_list = []
    standard_policy = []
    for gamma in gamma_range:
        ql = mdp.QLearning(P, R, gamma, n_iter=10e4)
        ql.run()

        timestr = "%0.3f" % (ql.time * 1000)
        atab = " \t"
        spacing = 3

        gamma_str = "%0.2f" % gamma
        msg = gamma_str + atab * spacing + timestr

        print(msg)
        if gamma == 0.95:
            standard_policy.append((ql.policy, mapping, shape))

        if list(ql.policy) == list(prev_policy):
            no_diff_list.append([prev_gamma, gamma])

        prev_policy = ql.policy
        prev_gamma = gamma
    print()
    print("Q Learning Iteration Policy at Gamma = 0.95")
    contents = standard_policy.pop()
    print_policy(contents[0], contents[1], contents[2])
    print()
    no_diff_len = len(no_diff_list)
    str_list = ["No Policy Difference Between These Gammas: "] * no_diff_len
    policy_diffs = zip(str_list, no_diff_list)
    for diff in policy_diffs:
        print("%s %0.2f %0.2f" % (diff[0], diff[1][0], diff[1][1]))
    print()


def frozen_lake_pim(P, R, gamma_range, mapping, shape):
    print("== Policy Modified Iteration ==")
    print("gamma    #Iterations     time (ms)")
    prev_policy = []
    prev_gamma = 0
    no_diff_list = []
    standard_policy = []
    for gamma in gamma_range:
        pim = mdp.PolicyIterationModified(P, R, gamma, max_iter=10000)
        pim.run()

        timestr = "%0.3f" % (pim.time * 1000)
        atab = " \t"
        if pim.iter <= 99:
            spacing = 4
        else:
            spacing = 3

        gamma_str = "%0.2f" % gamma
        msg = gamma_str + atab + str(pim.iter) + atab * spacing + timestr

        print(msg)
        if gamma == 0.95:
            standard_policy.append((pim.policy, mapping, shape))

        if list(pim.policy) == list(prev_policy):
            no_diff_list.append([prev_gamma, gamma])

        prev_policy = pim.policy
        prev_gamma = gamma
    print()
    print("Policy Modified Iteration Policy at Gamma = 0.95")
    contents = standard_policy.pop()
    print_policy(contents[0], contents[1], contents[2])
    print()
    no_diff_len = len(no_diff_list)
    str_list = ["No Policy Difference Between These Gammas: "] * no_diff_len
    policy_diffs = zip(str_list, no_diff_list)
    for diff in policy_diffs:
        print("%s %0.2f %0.2f" % (diff[0], diff[1][0], diff[1][1]))
    print()


def frozen_lake_pi(P, R, gamma_range, mapping, shape):
    print("== Policy Iteration ==")
    print("gamma    #Iterations     time (ms)")
    prev_policy = []
    prev_gamma = 0
    no_diff_list = []
    standard_policy = []
    for gamma in gamma_range:
        pi = mdp.PolicyIteration(P, R, gamma, max_iter=10000)
        pi.run()

        timestr = "%0.3f" % (pi.time * 1000)
        atab = " \t"
        if pi.iter <= 99:
            spacing = 4
        else:
            spacing = 3

        gamma_str = "%0.2f" % gamma
        msg = gamma_str + atab + str(pi.iter) + atab * spacing + timestr

        print(msg)
        if gamma == 0.95:
            standard_policy.append((pi.policy, mapping, shape))

        if list(pi.policy) == list(prev_policy):
            no_diff_list.append([prev_gamma, gamma])

        prev_policy = pi.policy
        prev_gamma = gamma
    print()
    print("Policy Iteration Policy at Gamma = 0.95")
    contents = standard_policy.pop()
    print_policy(contents[0], contents[1], contents[2])
    print()
    no_diff_len = len(no_diff_list)
    str_list = ["No Policy Difference Between These Gammas: "] * no_diff_len
    policy_diffs = zip(str_list, no_diff_list)
    for diff in policy_diffs:
        print("%s %0.2f %0.2f" % (diff[0], diff[1][0], diff[1][1]))
    print()


def frozen_lake_vi(P, R, gamma_range, mapping, shape):
    print("== Value Iteration == ")
    print("gamma    # Iterations    time (ms)")
    prev_policy = []
    prev_gamma = 0
    no_diff_list = []
    standard_policy = []

    for i, gamma in enumerate(gamma_range):
        vi = mdp.ValueIteration(P, R, gamma, max_iter=10000, epsilon=0.001)
        vi.run()

        timestr = "%0.3f" % (vi.time * 1000)
        atab = " \t"
        if vi.iter <= 99:
            spacing = 4
        else:
            spacing = 3

        gamma_str = "%0.2f" % gamma
        msg = gamma_str + atab + str(vi.iter) + atab * spacing + timestr

        print(msg)
        if gamma == 0.95:
            standard_policy.append((vi.policy, mapping, shape))

        if list(vi.policy) == list(prev_policy):
            no_diff_list.append([prev_gamma, gamma])

        prev_policy = vi.policy
        prev_gamma = gamma

    print()
    print("Value Iteration Policy at Gamma = 0.95")
    contents = standard_policy.pop()
    print_policy(contents[0], contents[1], contents[2])
    print()
    no_diff_len = len(no_diff_list)
    str_list = ["No Policy Difference Between These Gammas: "] * no_diff_len
    policy_diffs = zip(str_list, no_diff_list)
    for diff in policy_diffs:
        print("%s %0.2f %0.2f" % (diff[0], diff[1][0], diff[1][1]))
    print()


def evaluate_rewards_and_transitions(problem, mutate=False):

    # Enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # Intiailize P and R matrices
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_actions, num_states, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                if reward != 0.0:
                    reward *= 100.0
                else:
                    reward -= 0.01
                print(reward)
                P[action, state, next_state] = probability
                R[action, state, next_state] = reward

            # Normalize T across state + action axes
            P[action, state, :] /= np.sum(P[action, state, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.P = P
        problem.env.R = R

    return P, R


def frozen_pi_experiment(env_name, new_lake):
    np.random.seed(0)
    min_r = -100.0
    max_r = 100.0
    env = MyWrapper.TransformReward(gym.make(env_name, desc=new_lake), lambda r: np.clip(r*100.0, min_r, max_r))
    env.seed(0)
    env.reset()
    total_times = [0] * 10
    gammas = [0] * 10
    num_iterations = [0] * 10
    average_reward_list = [0] * 10

    for i in range(0, 10):
        start_time = time.time()
        policy_iter_instance = policy_iteration(env, (i + 0.5) / 10, 0.0001, 100000)
        improved_policy, value_vector, iteration_counter = (
            policy_iter_instance.policy_improvement()
        )
        average_improved_policy_reward = policy_iter_instance.policy_evaluation(
            improved_policy
        )  # average reward per iteration
        end_time = time.time()

        gammas[i] = (i + 0.5) / 10
        # print(iteration_counter)
        num_iterations[i] = iteration_counter
        total_times[i] = (end_time - start_time) * 1000  # in millisecond
        average_reward_list[i] = average_improved_policy_reward

    # for plotting, gamma vs reward,  gamma vs time, gamma vs iteration,  iteration vs reward, iteration vs computation time??
    """plot 1: gamma vs reward """
    plt.title("gamma_vs_reward")
    plt.plot(gammas, average_reward_list)
    plt.xlabel("gammas")
    plt.ylabel("average_reward_by_optimal_policy")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_PolicyIteration_gamma_vs_reward.png"
    )
    plt.close()
    plt.figure()
    #
    #
    """ plot2: gamma vs time"""
    plt.title("gamma_vs_iteration")
    plt.plot(gammas, num_iterations)  # in mili seconds
    plt.xlabel("gammas")
    plt.ylabel("num_iterations")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_PolicyIteration_gamma_vs_iteration.png"
    )
    plt.close()
    plt.figure()
    #
    """ plot3: gamma vs time"""
    plt.title("gamma_vs_time")
    plt.plot(gammas, total_times)  # in mili seconds
    plt.xlabel("gammas")
    plt.ylabel("computational time (mili seconds)")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_PolicyIteration_gamma_vs_time.png"
    )
    plt.close()
    plt.figure()
    #
    #
    #
    #
    """ plot4: iteration vs reward"""
    plt.title("iteration_vs_reward")
    plt.plot(
        num_iterations, average_reward_list
    )  # in mili seconds, iteration here is when its break
    plt.xlabel("num_iterations")
    plt.ylabel("average_reward_by_optimal_policy")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_PolicyIteration_iteration_vs_reward.png"
    )
    plt.close()
    plt.figure()
    #
    #
    """ plot5: iteration vs computation time"""
    plt.title("iteration_vs_time")
    plt.plot(
        num_iterations, total_times
    )  # in mili seconds, iteration here is when its break
    plt.xlabel("num_iterations")
    plt.ylabel("computational time (mili seconds)")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_PolicyIteration_iteration_vs_time.png"
    )
    plt.close()
    plt.figure()


def frozen_vi_experiment(env_name, new_lake):
    np.random.seed(0)
    min_r = -100.0
    max_r = 100.0
    env = MyWrapper.TransformReward(gym.make(env_name, desc=new_lake), lambda r: np.clip(r*100.0, min_r, max_r))
    env.seed(0)
    env.reset()

    # plotting, convergence comparison, time vs iteration,  reward vs iteration
    total_times = [0] * 10
    gammas = [0] * 10
    num_iterations = [0] * 10
    average_reward_list = [0] * 10

    for i in range(0, 10):
        start_time = time.time()
        new_instance = value_iteration(env, (i + 0.5) / 10, 0.0001, 100000)
        optimal_policy, value_vector, iteration_counter = new_instance.value_iteration()
        average_policy_reward = new_instance.policy_evaluation(
            optimal_policy
        )  # average reward per iteration
        end_time = time.time()

        gammas[i] = (i + 0.5) / 10
        num_iterations[i] = iteration_counter
        total_times[i] = (end_time - start_time) * 1000  # in mili second
        average_reward_list[i] = average_policy_reward

    # for plotting, gamma vs reward,  gamma vs time, gamma vs iteration,  iteration vs reward, iteration vs computation time??
    """plot 1: gamma vs reward """
    plt.title("gamma_vs_reward")
    plt.plot(gammas, average_reward_list)
    plt.xlabel("gammas")
    plt.ylabel("average_reward_by_optimal_policy")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_ValueIteration_gamma_vs_reward.png"
    )
    plt.close()
    plt.figure()

    """ plot2: gamma vs time"""
    plt.title("gamma_vs_iteration")
    plt.plot(gammas, num_iterations)  # in mili seconds
    plt.xlabel("gammas")
    plt.ylabel("num_iterations")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_ValueIteration_gamma_vs_iteration.png"
    )
    plt.close()
    plt.figure()

    """ plot3: gamma vs iteration"""
    plt.title("gamma_vs_time")
    plt.plot(gammas, total_times)  # in mili seconds
    plt.xlabel("gammas")
    plt.ylabel("computational time (mili seconds)")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_ValueIteration_gamma_vs_time.png"
    )
    plt.close()
    plt.figure()

    """ plot4: iteration vs reward"""
    plt.title("iteration_vs_reward")
    plt.plot(
        num_iterations, average_reward_list
    )  # in mili seconds, iteration here is when its break
    plt.xlabel("num_iterations")
    plt.ylabel("average_reward_by_optimal_policy")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_ValueIteration_iteration_vs_reward.png"
    )
    plt.close()
    plt.figure()

    """ plot5: iteration vs computation time"""
    plt.title("iteration_vs_time")
    plt.plot(
        num_iterations, total_times
    )  # in mili seconds, iteration here is when its break
    plt.xlabel("num_iterations")
    plt.ylabel("computational time (mili seconds)")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_ValueIteration_iteration_vs_time.png"
    )
    plt.close()
    plt.figure()
    pass


def frozen_ql_experiment(env_name, new_lake):
    np.random.seed(0)
    min_r = -100.0
    max_r = 100.0
    problem = MyWrapper.TransformReward(gym.make(env_name, desc=new_lake), lambda r: np.clip(r*100.0, min_r, max_r))
    problem.seed(0)
    problem.reset()
    folder = "q_learning/"
    env = MyWrapper.Monitor(problem, folder, force=True)
    # env.observation_space.n is number of states

    # q_table = np.zeros((env.observation_space.n, env.action_space.n)) # param -> q_table
    num_of_states = env.observation_space.n
    num_of_action = env.action_space.n
    rewards_list = []  # this will record reward for that run
    iterations_list = []  # this will record all number of iteration
    alpha = [0.5, 0.9]  # param -> alpha  [0.45, 0.65, 0.85] current 0.45
    gamma = 0.99  # param -> gamma
    episodes = 10000
    rar = [0.1, 0.9]  # epsilon [0.1,0.3,0.5,0.7,0.9], current 0.1
    radr = 0.99  # randomess decay
    time_list = []
    # begin the timer before the iteration begin

    # initialize the qlearner here
    qlearner = QLearner(
        num_actions=num_of_action,
        num_states=num_of_states,
        alpha=alpha[0],
        gamma=gamma,
        rar=rar[0],
        radr=radr,
    )
    # print(qlearner.q_table)

    """This is for plot #1 """
    # total time spend per episode
    init_time_diff = 0

    for episode in range(episodes):  # total number of iterations
        start_time = time.time()
        qlearner.s = env.reset()  # current state

        done = False
        total_reward = 0  # this is the initial reward i have
        max_steps = 10000000

        # print(state)
        for i in range(max_steps):
            if done:
                break
            # update qlearner.s by state
            """Key step, refer to the qlearner implementation """

            # update s before use as an input
            # action here is either a random action or the best action of the given state
            action = qlearner.choose_best_action(
                qlearner.num_actions, qlearner.rar, qlearner.s, qlearner.q_table
            )  # use current q_table
            # qlearner.s = qlearner.s
            # get state reward  done, info from the environment
            next_state, reward, done, info = env.step(action)  # this will update done
            # qlearner.s = qlearner.s  already updated
            qlearner.a = action
            # update my reward
            total_reward += reward
            """  right now the problem is that q table is not being updated"""
            # reward is current reward, total_reward is cumulative reward
            # update q-table on q[qlearner.s, action] using state(future_state) and reward,
            temp_action = qlearner.query(
                next_state, reward, False
            )  # this step will not update self.s and self.a
            # update state to next state, action is already updated, we good
            qlearner.s = next_state

        end_time = time.time()
        time_spend_one_episode = (end_time - start_time) * 1000
        init_time_diff += (
            time_spend_one_episode
        )  # by the end of iteration cumulative time

        time_list.append(init_time_diff)

        rewards_list.append(total_reward)  # total rewards for this episode
        iterations_list.append(
            i
        )  # record current iteration when it's done for the episide

    # close the environment,  find the time difference
    env.close()

    def chunk_list(l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    """rewards vs # of iterations plot"""
    episode_size = int(episodes / 50)
    segments = list(chunk_list(rewards_list, episode_size))
    average_reward = [sum(segment) / len(segment) for segment in segments]

    plt.title("Average Rewards vs Iterations (learning rate: 0.5, Epsilon: 0.1)")
    plt.plot(range(0, len(rewards_list), episode_size), average_reward)
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_qlearner_reward_vs_iterations.png"
    )
    plt.close()
    plt.figure()
    """plot 1 done """

    """Plot 2 computation time vs episodes """
    plt.title("Computation time vs episodes (learning rate: 0.5, Epsilon: 0.1)")
    plt.plot(range(0, episodes, 1), time_list)
    plt.xlabel("episodes")
    plt.ylabel("computation time (mili seconds)")
    plt.savefig("./plots/frozen_lake_experiment/computation_time_vs_episodes.png")
    plt.close()
    plt.figure()

    """This is for plot #3 change alpha:0.9, rar 0.1 """
    # plot 2 alpha = 0.65 vs reward
    single_alpha = alpha[1]  # alpha = 0.9
    rewards_list = []  # this will record reward for that run
    iterations_list = []  # this will record all number of iteration
    time_list = []
    init_time_diff = 0

    qlearner = QLearner(
        num_actions=num_of_action,
        num_states=num_of_states,
        alpha=single_alpha,
        gamma=gamma,
        rar=rar[0],
        radr=radr,
    )
    for episode in range(episodes):  # total number of iterations
        start_time = time.time()
        qlearner.s = env.reset()  # current state

        done = False
        total_reward = 0  # this is the initial reward i have
        max_steps = 10000000

        # print(state)
        for i in range(max_steps):
            if done:
                break
            # update qlearner.s by state
            """Key step, refer to the qlearner implementation """
            # start the timer

            # update s before use as an input
            # action here is either a random action or the best action of the given state
            action = qlearner.choose_best_action(
                qlearner.num_actions, qlearner.rar, qlearner.s, qlearner.q_table
            )  # use current q_table
            # qlearner.s = qlearner.s
            # get state reward  done, info from the environment
            next_state, reward, done, info = env.step(action)  # this will update done
            # qlearner.s = qlearner.s  already updated
            qlearner.a = action
            # update my reward
            total_reward += reward
            """  right now the problem is that q table is not being updated"""
            # reward is current reward, total_reward is cumulative reward
            # update q-table on q[qlearner.s, action] using state(future_state) and reward,
            temp_action = qlearner.query(
                next_state, reward, False
            )  # this step will not update self.s and self.a
            # update state to next state, action is already updated, we good
            qlearner.s = next_state

        end_time = time.time()
        time_spend_one_episode = (end_time - start_time) * 1000
        init_time_diff += (
            time_spend_one_episode
        )  # by the end of iteration cumulative time
        time_list.append(init_time_diff)

        rewards_list.append(total_reward)  # total rewards for this episode
        iterations_list.append(
            i
        )  # record current iteration when it's done for the episide

    # close the environment,  find the time difference

    """plot 3"""
    episode_size = int(episodes / 50)
    segments = list(chunk_list(rewards_list, episode_size))
    average_reward = [sum(segment) / len(segment) for segment in segments]

    plt.title("Reward vs Iteration (Learning Rate: 0.9, Epsilon:0.1)")
    # print(single_alpha)
    plt.plot(range(0, len(rewards_list), episode_size), average_reward)
    plt.xlabel("Iterations")
    plt.ylabel("Average Rewards")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_qlearner_rewards_vs_iter_alpha0.9.png"
    )
    plt.close()
    plt.figure()

    """plot 4 time vs iters"""
    plt.title("Computation time vs episodes (learning rate: 0.9, Epsilon: 0.1)")
    plt.plot(range(0, episodes, 1), time_list)
    plt.xlabel("episodes")
    plt.ylabel("computation time (mili seconds)")
    plt.savefig(
        "./plots/frozen_lake_experiment/computation_time_vs_episodes_alpha0.9.png"
    )
    plt.close()
    plt.figure()

    """This is for plot #4  alpha: 0.5, rar(epsilon) 0.9"""
    single_alpha = alpha[0]  # alpha = 0.9
    single_rar = rar[1]
    rewards_list = []  # this will record reward for that run
    iterations_list = []  # this will record all number of iteration
    time_list = []
    init_time_diff = 0

    qlearner = QLearner(
        num_actions=num_of_action,
        num_states=num_of_states,
        alpha=single_alpha,
        gamma=gamma,
        rar=single_rar,
        radr=radr,
    )
    for episode in range(episodes):  # total number of iterations
        start_time = time.time()
        qlearner.s = env.reset()  # current state

        done = False
        total_reward = 0  # this is the initial reward i have
        max_steps = 10000

        # print(state)
        for i in range(max_steps):
            if done:
                break
            # update qlearner.s by state
            """Key step, refer to the qlearner implementation """
            # start the timer

            # update s before use as an input
            # action here is either a random action or the best action of the given state
            action = qlearner.choose_best_action(
                qlearner.num_actions, qlearner.rar, qlearner.s, qlearner.q_table
            )  # use current q_table
            # qlearner.s = qlearner.s
            # get state reward  done, info from the environment
            next_state, reward, done, info = env.step(action)  # this will update done
            # qlearner.s = qlearner.s  already updated
            qlearner.a = action
            # update my reward
            total_reward += reward
            """  right now the problem is that q table is not being updated"""
            # reward is current reward, total_reward is cumulative reward
            # update q-table on q[qlearner.s, action] using state(future_state) and reward,
            temp_action = qlearner.query(
                next_state, reward, False
            )  # this step will not update self.s and self.a
            # update state to next state, action is already updated, we good
            qlearner.s = next_state

        end_time = time.time()
        time_spend_one_episode = (end_time - start_time) * 1000
        init_time_diff += (
            time_spend_one_episode
        )  # by the end of iteration cumulative time
        time_list.append(init_time_diff)

        rewards_list.append(total_reward)  # total rewards for this episode
        iterations_list.append(
            i
        )  # record current iteration when it's done for the episide

    """plot 5 reward vs iteration"""
    episode_size = int(episodes / 50)
    segments = list(chunk_list(rewards_list, episode_size))
    average_reward = [sum(segment) / len(segment) for segment in segments]

    plt.title("Reward vs Iteration (Learning Rate: 0.5, Epsilon:0.9)")
    # print(single_alpha)
    plt.plot(range(0, len(rewards_list), episode_size), average_reward)
    plt.xlabel("Iterations")
    plt.ylabel("Average Rewards")
    plt.savefig(
        "./plots/frozen_lake_experiment/frozen_qlearner_rewards_vs_iter_epsilon0.9.png"
    )
    plt.close()
    plt.figure()

    """plot 6 time vs iters"""
    plt.title("Computation time vs episodes (learning rate: 0.5, Epsilon: 0.9)")
    plt.plot(range(0, episodes, 1), time_list)
    plt.xlabel("episodes")
    plt.ylabel("computation time (mili seconds)")
    plt.savefig(
        "./plots/frozen_lake_experiment/computation_time_vs_episodes_epsilon0.9.png"
    )
    plt.close()
    plt.figure()


if __name__ == "__main__":
    # mdp_example()
    # print()

    # Frozen Lake Small
    # mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
    # shape = (4, 4)
    # run_gridworld("FrozenLake-v0", mapping, shape)

    # Frozen Lake Large
    n = 16
    shape = (n, n)
    mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
    np.random.seed(0)
    new_frozenlake = generate_random_map(n, 0.98046875)
    run_gridworld("FrozenLake-v0", mapping, shape, new_frozenlake)

    frozen_vi_experiment("FrozenLake-v0", new_frozenlake)
    frozen_pi_experiment("FrozenLake-v0", new_frozenlake)
    frozen_ql_experiment("FrozenLake-v0", new_frozenlake)
