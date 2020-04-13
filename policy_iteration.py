import numpy as np
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt


# adapt from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
class policy_iteration(
    object):  # the goal is to find the optimal policy, that maximizes value function of the environment

    def __init__(self, env, gamma, threshold, max_iteration):
        """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    Returns:
        Vector of length env.nS representing the value function.
    """

        self.env = env
        self.gamma = gamma
        self.threshold = threshold  # this is also known as theta the threshold
        self.max_iteration = max_iteration

    '''This method is done '''

    def internal_policy_evaluation(self, policy):

        env = self.env
        threshold = self.threshold  # this is the threshold theta
        gamma = self.gamma  # this is the discount rate

        # Start with a random (all 0) value function
        V = np.zeros(env.nS)  # V is the same size as number of state
        while True:
            delta_value = 0  # initialize current difference or delta_value to 0
            for state in range(env.nS):  # iterate throguh every state
                v = 0  # initialize smalle v
                for action, action_probability in enumerate(policy[state]):  # iterate all action value and action index

                    for probability, next_state, reward, done in env.P[state][
                        action]:  # P[state][action] will return an aciton value
                        # Calculate the expected value
                        v += action_probability * probability * (reward + gamma * V[next_state])
                # How much our value function changed (across any states)
                delta_value = max(delta_value, np.abs(v - V[state]))
                V[state] = v
            # Stop evaluating once our value function change is below a threshold
            if delta_value < threshold:
                break
        return np.array(V)  # return the V vector

    '''This method is done   *****'''

    def policy_improvement(self):

        env = self.env
        gamma = self.gamma  # -> this is the discount rate
        max_iteration = self.max_iteration

        # Start with a random policy
        policy = np.ones([env.nS, env.nA]) / env.nA  # this is the random policy
        counter = 0
        for i in range(max_iteration):

            V = self.internal_policy_evaluation(policy)  # evaluate the random policy
            policy_stable = True  # if change are made to policy will update to false

            # For each state...

            for state in range(env.nS):  # iterate through every state
                # The best action we would take under the current policy
                chosen_action = np.argmax(policy[state])  # choose the action that has best action

                action_values = self.util_compute_action_value(state, V)  # evaluate and find best action value vector
                best_action = np.argmax(action_values)  # return the index position of the best action values

                if chosen_action != best_action:  # if chosen action is not the same as the best action
                    policy_stable = False
                    # policy is not stable
                # print(np.eye(env.nA)[best_action])
                policy[state] = np.eye(env.nA)[best_action]

            if policy_stable:  # meaning if both actions are equal, we are done, return the counter
                counter = i + 1  # counter equal to current iteration + 1
                return policy, V, counter  # this is known as policy iteration

    '''This method is good to go '''

    def util_compute_action_value(self, current_state, V):
        '''This method helps calculate value for all aciton in a given state '''
        '''
        Input:
        1) current_state: state I am evaluating on
        2) V: vector to store value of each state
        Output:
        Action vector that computes value of each possible actions
        '''
        env = self.env

        A = np.zeros(env.nA)
        for action in range(env.nA):
            for probability, next_state, reward, done in env.P[current_state][action]:
                A[action] += probability * (reward + self.gamma * V[next_state])
        return A  # A here is the value for action vector

    '''This method is ready to go '''

    def policy_evaluation(self, policy, n=100):  # original input: policy, env, gamma, n = 100
        '''This function evaluates how well a specific policy is based on it's average reward '''
        # need to use get_reward_for_episode to obtain score
        '''
        input:
        1) env: environment
        2) policy: best policy from value_iteration
        3) gamma: same gamma as before
        Output:
        score: mean policy score
        '''
        env = self.env
        gamma = self.gamma

        score = [self.util_calculate_episode_total_reward(policy, render=False) for i in range(n)]

        return np.mean(score)

    '''This method is ready to go '''

    def util_calculate_episode_total_reward(self, policy,
                                            render=False):  # ? utility function that return total reward per episode

        env = self.env
        gamma = self.gamma
        observation = env.reset()
        # print("observation is {}".format(observation))
        # print("kw test 1")
        # print("current_observation is {}".format(observation))
        cumulative_reward = 0
        index_step = 0
        # print("kw test 2")
        # print("policy is {}".format(policy))
        # print("max_policy is {}".format(policy[0]))
        # print("kw test 3")
        while True:
            if render:
                env.render()
            new_observation, current_reward, done, info = env.step(
                np.argmax(policy[observation]))  # it could be potential error here
            cumulative_reward += (gamma ** index_step * current_reward)
            index_step += 1

            if done:
                break
        return cumulative_reward  # this is the total reward for this particula episide, consult from the action taken