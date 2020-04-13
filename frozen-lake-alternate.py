import gym
import numpy as np

from hiive.mdptoolbox.mdp import (
    ValueIteration,
    QLearning,
    PolicyIteration,
)
from gym.envs.toy_text.frozen_lake import generate_random_map

import matplotlib.pyplot as plt


def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


if __name__ == "__main__":
    n = 16
    np.random.seed(0)
    new_frozenlake = generate_random_map(n, 0.9)
    mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
    shape = (n, n)

    env = gym.make("FrozenLake-v0", desc=new_frozenlake)

    Gamma = 0.99

    env.reset()

    # Enumerate state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))
    # prepare gym for mdptoolbox
    for state in env.env.P:
        for action in env.env.P[state]:
            for option in env.env.P[state][action]:
                P[action][state][option[1]] += option[0]
                R[state][action] += option[2]

    vi = ValueIteration(P, R, Gamma, epsilon=0.01, max_iter=20000)

    # run vi
    vi.setVerbose()
    vi.run()
    print("== Value Iteration ==")
    print("Policy: ")
    print_policy(vi.policy, mapping, shape)
    print(vi.policy)
    print("Iterations: ")
    print(vi.iter)
    print("Time: ")
    print(vi.time)
    print(vi.run_stats[-1:])

    iterations = np.zeros(len(vi.run_stats))
    reward = np.zeros(len(vi.run_stats))
    i = 0
    for stat in vi.run_stats:
        iterations[i] = stat["Iteration"]
        reward[i] = stat["Reward"]
        i += 1

    fig, ax = plt.subplots()
    ax.plot(iterations, reward)

    ax.set(xlabel="Iterations", ylabel="Reward", title="Frozen Lake Value Iteration")
    ax.grid()

    fig.savefig("frozen-lake.vi.png")

    pi = PolicyIteration(P, R, Gamma, None, max_iter=20000)

    # run pi
    pi.setVerbose()
    pi.run()
    print("== Policy Iteration ==")
    print("Policy: ")
    print_policy(pi.policy, mapping, shape)
    print("Iterations: ")
    print(pi.iter)
    print("Time: ")
    print(pi.time)
    print(pi.run_stats[-1:])

    iterations = np.zeros(len(pi.run_stats))
    reward = np.zeros(len(pi.run_stats))
    i = 0
    for stat in pi.run_stats:
        iterations[i] = stat["Iteration"]
        reward[i] = stat["Reward"]
        i += 1

    fig, ax = plt.subplots()
    ax.plot(iterations, reward)

    ax.set(xlabel="Iterations", ylabel="Reward", title="Frozen Lake Policy Iteration")
    ax.grid()

    fig.savefig("frozen-lake.pi.png")

    print("== Q Learning ==")
    values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.99]
    resultRewards = [None] * len(values)
    resultIterations = [None] * len(values)
    i = 0
    for v in values:
        QL = QLearning(
            P, R, Gamma, n_iter=10000000, epsilon=0.1, epsilon_decay=v, epsilon_min=0.1
        )
        # run QL
        QL.setVerbose()
        QL.run()
        print("QL")
        print(QL.time)
        print(QL.run_stats[-1:])

        resultIterations[i] = np.zeros(len(QL.run_stats))
        resultRewards[i] = np.zeros(len(QL.run_stats))
        j = 0
        sum = 0
        for stat in QL.run_stats:
            sum += stat["Reward"]
            resultIterations[i][j] = stat["Iteration"]
            resultRewards[i][j] = sum
            j += 1

        i += 1

    fig, ax = plt.subplots()

    for i in range(len(values)):
        ax.plot(resultIterations[i], resultRewards[i], label=values[i])

    ax.set(
        xlabel="Iterations", ylabel="Accumulated Reward", title="Frozen Lake Q-Learning"
    )
    ax.grid()
    ax.legend()
    fig.savefig("frozen-lake.ql.decay.png")

    QL = QLearning(P, R, Gamma, n_iter=10000000, epsilon_decay=0.9)
    # run QL
    QL.setVerbose()
    QL.run()
    print("Policy: ")
    print(QL.policy)
    print_policy(QL.policy, mapping, shape)
    print("Time: ")
    print(QL.time)
    print("Mean Discrepancy: ")
    print(QL.error_mean)
    # print(QL.v_mean)
    print("Epsilon: ")
    print(QL.epsilon)
    print(QL.run_stats[-1:])

    iterations = np.zeros(len(QL.run_stats))
    reward = np.zeros(len(QL.run_stats))
    i = 0
    sum = 0
    for stat in QL.run_stats:
        sum += stat["Reward"]
        iterations[i] = stat["Iteration"]
        reward[i] = sum
        i += 1

    fig, ax = plt.subplots()
    ax.plot(iterations, reward)

    ax.set(
        xlabel="Iterations",
        ylabel="Accumulated Reward",
        title="Frozen Lake Q-Learning epsilon_decay=0.9",
    )
    ax.grid()
    fig.savefig("frozen-lake.ql.png")

    print(QL.policy == vi.policy)
    print(vi.policy == pi.policy)
    print(vi.policy)
    print(pi.policy)
    print(QL.policy)
