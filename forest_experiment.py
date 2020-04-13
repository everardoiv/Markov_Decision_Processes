from hiive import mdptoolbox
from hiive.mdptoolbox import example, mdp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def timing(func):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = func(*args, **kwargs)
        time2 = time.time()
        print("%s function took %0.3f ms" % (func.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


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


def plot_stats(stats, name, y_col='Error'):
    df = pd.DataFrame.from_records(stats)
    print(df.tail(5))
    plt.clf()
    plt.title(('%s, time - %0.3f' % (name, df['Time'].max())))
    plt.xlabel('Iterations')
    plt.ylabel(y_col)
    df.plot(x='Iteration', y=y_col, kind='line')
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/single_episode/%s_%s.png' % (name, y_col))

    plt.clf()
    plt.title(('%s, time - %0.3f' % (name, df['Time'].sum())))
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    df.plot(x='Iteration', y=y_col, kind='line')
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/single_episode/reward_v_iteration_%s.png' % (name))

    plt.clf()
    plt.title('Iterations v Time')
    plt.xlabel('Time')
    plt.ylabel('Iterations')
    df.plot(x='Time', y='Iteration', kind='line')
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/single_episode/iterations_v_time_%s.png' % name)

    plt.close('all')


def run_forest():
    np.random.seed(0)
    P, R = example.forest(S=5, r1=3, r2=15, p=0.2)
    print("Transition Array: ")
    print(P.shape)
    print(P)  # Transition array A x S x S
    print("Reward Array: ")
    print(R.shape)
    print(R)  # Reward array S x A

    # TODO
    gamma_range = np.array([0.1, 0.9, 0.99])
    alpha_range = np.array([0.01, 0.5, 0.99])
    epsilon_range = np.array([0.1, 0.5, 0.95])
    e_decay_range = np.array([0.1, 0.5, 0.999])

    # gamma_range = np.append(np.linspace(0.1, 0.9, 9), np.linspace(0.91, 0.99, 9))
    # alpha_range = np.append(np.linspace(0.01, 0.1, 9), np.linspace(0.2, 0.99, 4))
    # epsilon_range = np.linspace(0.1, 1.0, 10)
    # e_decay_range = np.append(np.linspace(0.1, 0.9, 4), np.linspace(0.91, 0.99, 9))

    difference_list = np.zeros(gamma_range.shape)
    value_iteration_list = np.zeros(gamma_range.shape)
    value_time_list = np.zeros(gamma_range.shape)
    value_reward_list = np.zeros(gamma_range.shape)
    value_error_list = np.zeros(gamma_range.shape)

    policy_iteration_list = np.zeros(gamma_range.shape)
    policy_time_list = np.zeros(gamma_range.shape)
    policy_reward_list = np.zeros(gamma_range.shape)
    policy_error_list = np.zeros(gamma_range.shape)

    for i, gamma in enumerate(gamma_range):
        print('Gamma %0.2f' % gamma)

        vi = mdp.ValueIteration(transitions=P, reward=R, gamma=gamma, epsilon=0.0001, max_iter=10000)
        vi.setVerbose()
        vi.run()
        vi_stats = vi.run_stats
        value_iteration_list[i] = vi_stats[-1:][0]['Iteration']
        value_time_list[i] = vi_stats[-1:][0]['Time']
        value_reward_list[i] = vi_stats[-1:][0]['Reward']
        value_error_list[i] = vi_stats[-1:][0]['Error']
        plot_stats(vi_stats, ('vi_forest_%0.2f' % gamma))

        pi = mdp.PolicyIteration(transitions=P, reward=R, gamma=gamma, max_iter=10000, eval_type=1)
        pi.setVerbose()
        pi.run()
        stats = pi.run_stats
        policy_iteration_list[i] = stats[-1:][0]['Iteration']
        policy_time_list[i] = stats[-1:][0]['Time']
        policy_reward_list[i] = stats[-1:][0]['Reward']
        policy_error_list[i] = stats[-1:][0]['Error']
        plot_stats(stats, ('pi_forest_%0.2f' % gamma))
        print('Policies Found')
        print('Value Iteration: ' + str(vi.policy))
        print('Policy Iteration: ' + str(pi.policy))

        difference1 = sum([abs(x - y) for x, y in zip(pi.policy, vi.policy)])
        difference_list[i] = difference1
        print("Discrepancy in Policy and Value Iteration: ", difference1)
        print()

    # Plotting
    # Error v Iteration
    plt.clf()
    plt.title('Value Iteration: Error v Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.plot(list(value_iteration_list), list(value_error_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/vi_error_v_iteration.png')

    # Reward v Gamma
    plt.clf()
    plt.title('Value Iteration: Reward v Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Reward')
    plt.plot(list(gamma_range), list(value_reward_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/vi_reward_v_gamma.png')

    # Gamma v Iterations
    plt.clf()
    plt.title('Value Iteration: Gamma v Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Gamma')
    plt.plot(list(value_iteration_list), list(gamma_range))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/vi_gamma_v_iterations.png')

    # Gamma v Time
    plt.clf()
    plt.title('Value Iteration: Gamma v Time')
    plt.xlabel('Time')
    plt.ylabel('Gamma')
    plt.plot(list(value_time_list), list(gamma_range))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/vi_gamma_v_time.png')

    # Reward vs Iterations
    plt.clf()
    plt.title('Value Iteration: Reward v Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.plot(list(value_iteration_list), list(value_reward_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/vi_reward_v_iterations.png')

    # Policy
    # Error v Iteration
    plt.clf()
    plt.title('Policy Iteration: Error v Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.plot(list(policy_iteration_list), list(policy_error_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/pi_error_v_iteration.png')

    # Gamma v Reward
    plt.clf()
    plt.title('Policy Iteration: Reward v Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Reward')
    plt.plot(list(gamma_range), list(policy_reward_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/pi_reward_v_gamma.png')

    # Gamma v Iterations
    plt.clf()
    plt.title('Policy Iteration: Gamma v Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Gamma')
    plt.plot(list(policy_iteration_list), list(gamma_range))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/pi_gamma_v_iterations.png')

    # Gamma v Time
    plt.clf()
    plt.title('Policy Iteration: Gamma v Time')
    plt.xlabel('Time')
    plt.ylabel('Gamma')
    plt.plot(list(policy_time_list), list(gamma_range))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/pi_gamma_v_time.png')

    # Reward vs Iterations
    plt.clf()
    plt.title('Policy Iteration: Reward v Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.plot(list(policy_iteration_list), list(policy_reward_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/pi_reward_v_iterations.png')

    # Gamma vs Policy Differences
    plt.clf()
    plt.title('Gamma v Policy Differences')
    plt.xlabel('Gamma')
    plt.ylabel('Policy Differences')
    plt.plot(list(gamma_range), list(difference_list))
    plt.tight_layout()
    plt.savefig('plots/forest_experiment/gamma_v_differences.png')
    plt.close('all')

    prev_Q = None
    thresh = 1e-4
    print('== Q Learning ==')
    for i, gamma in enumerate(gamma_range):
        for j, alpha in enumerate(alpha_range):
            for k, ep in enumerate(epsilon_range):
                for l, ed in enumerate(e_decay_range):
                    # print('ql: gamma - {}, alpha - {}, epsilon - {}, e_decay - {}'.format(gamma, alpha, ep, ed))
                    ql = mdp.QLearning(transitions=P, reward=R, gamma=gamma, alpha=alpha, alpha_decay=1.0, alpha_min=0.001,
                                       epsilon=ep, epsilon_min=0.1, epsilon_decay=ed, n_iter=10e4)
                    stats = ql.run()
                    plot_stats(stats, ('ql_forest_%0.2f_%0.2f_%0.2f_%0.2f' % (gamma, alpha, ep, ed)))

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
                        if np.all(res < thresh) or variation < thresh or windowed_reward > 1.0:
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

    # Iteration v Reward



if __name__ == "__main__":
    mdp_example()
    print()
    run_forest()
