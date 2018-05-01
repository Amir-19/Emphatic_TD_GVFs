# first test on GTD(lambda)
# states from 0 -> 9
# the bhv policy is choosing random action (up(2)-stay(1)-down(0)) with equalprob
# the target policy is learning how many time steps is to go to state (0) if we are always going down
# we use GTD lambda and ETD lambda to learn this off-policy estimate

# TODO: plot the second set of weights! (magnitude of TD and correction and their direction shows many things :p
import numpy as np
from dynamic_plotter import *
from etd import *
from gtd import *

def next_state(state):

    a = np.random.choice([0,1,2])
    if state == 0:
        reward = 0
    else:
        reward = 1
    new_state = state + a - 1
    if (new_state > 9):
        new_state = 9
    elif (new_state < 0):
        new_state = 0
    return new_state, a, reward

def gamma(state):
    if state == 0:
        return 0
    return 1

def feature_vector(state):
    fvector = np.zeros(10)
    fvector[state] = 1.0
    return fvector


def experiment_off_policy():

    plotting = True
    if plotting:
        d = DynamicPlot(window_x = 100, title = 'Off-Policy Predictions', xlabel = 'Time_Step', ylabel= 'Value')
        d.add_line('Prediction ETD')
        d.add_line('Prediction GTD')
        d.add_line('State')

    # init problem
    num_state = 10
    num_action = 3

    # divide by the number of active features in the feature vector!
    alpha = 0.1/1
    beta = 0.01/1
    lam = 0.9

    # init state, action, and time step
    state = 5
    action = None
    t = 0

    # ETD and GTD algorithms
    etd_algo = ETD(num_state)
    gtd_algo = GTD(num_state)

    while True:

        state_prime, action, reward = next_state(state)

        if (action == 0):
            rho = 3
        else:
            rho = 0

        etd_algo.update(feature_vector(state),reward,feature_vector(state_prime),alpha,gamma(state),gamma(state_prime),lam,rho,1)
        gtd_algo.update(feature_vector(state),reward,feature_vector(state_prime),alpha,beta,gamma(state),gamma(state_prime),lam,lam,rho)

        if plotting:
            d.update(t,[etd_algo.get_value(feature_vector(state)),gtd_algo.get_value(feature_vector(state)),state])

        # go to the next step
        state = state_prime
        t += 1

experiment_off_policy()