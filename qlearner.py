import numpy as np
import random as rand


class QLearner(object):
    def __init__(
        self,
        num_actions=4,
        s=0,  # default state
        a=0,  # default action
        num_states=100,
        alpha=0.2,
        gamma=0.9,
        rar=0.4,
        radr=0.99,
    ):

        self.num_actions = num_actions
        self.s = s  # current s, initialize as 0 could be anything
        self.a = a  # current a, initialize as 0 could be anything
        self.num_states = num_states
        self.alpha = (
            alpha
        )  # alpha is the learning rate, the bigger the alpha, the faster it learn
        self.gamma = (
            gamma
        )  # gamma, is the discount rate, ranges from 0 - 1, high value gamma mean we value later rewards significantly
        self.rar = rar
        self.radr = radr
        self.q_table = np.zeros(
            (self.num_states, self.num_actions)
        )  # q_table should be an attribute of the class
        # self.T_table = np.zeros((self.num_states, self.num_actions, self.num_states)) # this represent the probability of state s take action a end up with s_prime
        # self.T_count_table  = np.full((self.num_states, self.num_actions, self.num_states), 0.00001)
        # self.R_table = np.zeros((self.num_states, self.num_actions))
        # self.experience_list = np.zeros((0,4))

    def choose_best_action(self, num_actions, rar, state, q_table):
        if (
            np.random.uniform() < rar
        ):  # if the randomness is greater than rar,pick a random actions
            action = np.random.randint(0, num_actions - 1)
            # print(action)
        else:
            action = np.argmax(
                q_table[state, :]
            )  # other wise pick action index with highest q val
            # else:
            #     action = np.argmax(q_table[state,:], axis = 1)

        return action

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions - 1)
        # update action, can choose random action sometimt
        action = self.choose_best_action(
            self.num_actions, self.rar, self.s, self.q_table
        )

        self.a = action

        return action  # input a state, get the best action

    def query(
        self, s_prime, r, need_updated
    ):  # input the next state and reward, return the best action
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # print self.s
        # print self.a
        # action = rand.randint(0, self.num_actions-1)
        """My code """

        # we execute normal q-learning
        improved_estimate = r + self.gamma * (
            self.q_table[
                s_prime,
                self.choose_best_action(
                    self.num_actions, self.rar, s_prime, self.q_table
                ),
            ]
        )
        self.q_table[self.s, self.a] = (1 - self.alpha) * self.q_table[
            self.s, self.a
        ] + self.alpha * improved_estimate
        action = self.choose_best_action(
            self.num_actions, self.rar, s_prime, self.q_table
        )

        # after setting the action update rar, update s to s', update a to new action

        if need_updated == True:
            self.rar = self.rar * self.radr
            self.s = s_prime  # current state equal future state
            # self.a = action # current action become future next best action
        """ my code """

        # print(self.q_table[self.s, self.a])

        return action  # last but not least return the best action


#
#
# if __name__=="__main__":
#     print "Remember Q from Star Trek? Well, this isn't him"
