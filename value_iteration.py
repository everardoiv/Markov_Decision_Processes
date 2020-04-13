import numpy as np
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt


'''Value Iteration Algorithm.
Args:
    env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        env.nS is a number of states in the environment.
        env.nA is a number of actions in the environment.
    theta: We stop evaluation once our value function change is less than theta for all states.
    discount_factor: Gamma discount factor.
Returns:
    A tuple (policy, V) of the optimal policy and the optimal value function.
'''
#adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
# this is a value iteration object, call this object in the experiment to plot
class value_iteration(object): # the goal is to find the optimal policy, that maximizes value function of the environment

    def __init__(self, env, gamma, threshold, max_iteration):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold  # this is also known as theta the threshold
        self.max_iteration = max_iteration

    def value_iteration(self): # original input: env, gamma, threshold
        '''
        Input:
        1) env: environment
        2) gamma: discount factor
        3) threshold: stop evaluation once our value function change is less than theta for all states (threshold)
        Output:
        optimal policy (p)
        Value function (V)
        '''
        env = self.env
        gamma = self.gamma
        threshold = self.threshold
        max_iteration = self.max_iteration

        V = np.zeros(env.nS)  # Value vector

        for i in range(max_iteration):
            counter = 0
            # stop
            val_delta = 0 # max difference differen the aciton taken(delta)
            for state in range(env.nS):  # for number in the range of number of states in the environment.
                A =  self.util_compute_action_value (state,V)                        # geth the action value of each state
                best_a_val = np.max(A)                 # store the best action value
                #now update delta, the maxium difference across all state
                val_delta = max(val_delta, np.abs(best_a_val - V[state])) #update delta by the max distance
                # also update the value vector of current state, by the action value
                V[state] = best_a_val

            if val_delta < threshold: # if the after check all state and the max_
                counter = i+1
                break

        # this is a deterministic policy that mean
        p =np.zeros([env.nS, env.nA])       # ??is this right? policy initialize policy matrix size [n_state, n_action] with all zeros
        for s in range(env.nS):
            # from state position 0 to total num, find th ebest action
            A = self.util_compute_action_value(s, V)  # return Action vector with size env.nA
            best_available_action = np.argmax(A)   # return position of the best action vecto
            #now updated the policy matrix based on this action posit
            p[s, best_available_action] =  1 # hard code the policy value at state s and action

        return p, V, counter  # policy( s by a matrix) and Value vector (state length vector)


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


    def policy_evaluation(self, policy, n= 100): # original input: policy, env, gamma, n = 100
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

        score = [self.util_calculate_episode_total_reward(policy,render = False) for i in range (n)]

        return np.mean(score)


    def util_calculate_episode_total_reward(self, policy,render = False): # ? utility function that return total reward per episode

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

            new_observation, current_reward, done, info = env.step(np.argmax(policy[observation])) # it could be potential error here
            cumulative_reward += (gamma ** index_step * current_reward)
            index_step += 1

            if done:
                break
        return cumulative_reward  # this is the total reward for this particula episide, consult from the action taken