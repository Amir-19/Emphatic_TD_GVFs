import numpy as np
from lib_robotis_hack import *
from dynamic_plotter import *
import thread
import time
import numpy as np
import signal
import utils
from etd import *
from gtd import *
import copy

# PROBLEM SETUP FROM: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/Baird.py
# all states: state 0-5 are upper states
STATES = np.arange(0, 7)
# state 6 is lower state
LOWER_STATE = 6
# discount factor
DISCOUNT = 0.99

# each state is represented by a vector of length 8
FEATURE_SIZE = 8
FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
for i in range(LOWER_STATE):
    FEATURES[i, i] = 2
    FEATURES[i, 7] = 1
FEATURES[LOWER_STATE, 6] = 1
FEATURES[LOWER_STATE, 7] = 2

# all possible actions
DASHED = 0
SOLID = 1
ACTIONS = [DASHED, SOLID]

# reward is always zero
REWARD = 0



# state distribution for the behavior policy
stateDistribution = np.ones(len(STATES)) / 7
stateDistributionMat = np.matrix(np.diag(stateDistribution))
# projection matrix for minimize MSVE
projectionMatrix = np.matrix(FEATURES) * \
                   np.linalg.pinv(np.matrix(FEATURES.T) * stateDistributionMat * np.matrix(FEATURES)) * \
                   np.matrix(FEATURES.T) * \
                   stateDistributionMat

# behavior policy
BEHAVIOR_SOLID_PROBABILITY = 1.0 / 7

# TO here for the problem setup from: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/Baird.py
# saving data variables
servo1_data = None
gtd_data = None
etd_data = None
flag_stop = False


# bin setting
first_bin = 0
last_bin = 7


def behaviorPolicy(state):
    if np.random.binomial(1, BEHAVIOR_SOLID_PROBABILITY) == 1:
        return SOLID
    return DASHED

def read_data(servo):
    read_all = [0x02, 0x24, 0x08]
    data = servo.send_instruction(read_all, servo.servo_id)
    return utils.parse_data(data)

# take @action at @state, return the new state
def takeAction(state, action,servo):
    if action == SOLID:
        servo.move_angle(1.3,blocking=True)
    go_to = np.random.choice(STATES[: LOWER_STATE])
    if go_to == 0:
        servo.move_angle(-1.35,blocking=True)
    elif go_to == 1:
        servo.move_angle(-0.85,blocking=True)
    elif go_to == 2:
        servo.move_angle(-0.3,blocking=True)
    elif go_to == 3:
        servo.move_angle(0.0,blocking=True)
    elif go_to == 4:
        servo.move_angle(0.45,blocking=True)
    elif go_to == 5:
        servo.move_angle(0.85,blocking=True)

# target policy
def targetPolicy(state):
    return SOLID

def get_angle_bin(ang,bins):

    ang_f_bin = ang + 1.5
    return np.digitize(ang_f_bin, bins)

# taken from: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/Baird.py
# compute RMSVE for a value function parameterized by @theta
# true value function is always 0 in this example
def computeRMSVE(theta):
    return np.sqrt(np.dot(np.power(np.dot(FEATURES, theta), 2), stateDistribution))

# taken from: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter11/Baird.py
# compute RMSPBE for a value function parameterized by @theta
# true value function is always 0 in this example
def computeRMSPBE(theta):
    bellmanError = np.zeros(len(STATES))
    for state in STATES:
        for nextState in STATES:
            if nextState == LOWER_STATE:
                bellmanError[state] += REWARD + DISCOUNT * np.dot(theta, FEATURES[nextState, :]) - np.dot(theta, FEATURES[state, :])
    bellmanError = np.dot(np.asarray(projectionMatrix), bellmanError)
    return np.sqrt(np.dot(np.power(bellmanError, 2), stateDistribution))


def gamma(state):
    return 0.99

def cummlant(state):
    return 0

def get_rho(state,action):
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    return rho

def feature_vector(state):
    return FEATURES[state, :]


def main():

    global servo1_data, flag_stop, gtd_data, etd_data
    servo1_data = []
    gtd_data = []
    etd_data = []

    # servo connection step
    D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QD8V", baudrate=1000000)
    s1 = Robotis_Servo(D, 2)
    s1.move_angle(-1.35,blocking=True)

    #plotting
    d1 = DynamicPlot(window_x=100, title='GTD Weights and Error', xlabel='Steps', ylabel='Value')
    d1.add_line('w1')
    d1.add_line('w2')
    d1.add_line('w3')
    d1.add_line('w4')
    d1.add_line('w5')
    d1.add_line('w6')
    d1.add_line('w7')
    d1.add_line('w8')
    d1.add_line('RMSVE')
    d1.add_line('RMSPBE')

    d2 = DynamicPlot(window_x=100, title='ETD Weights and Error', xlabel='Steps', ylabel='Value')
    d2.add_line('w1')
    d2.add_line('w2')
    d2.add_line('w3')
    d2.add_line('w4')
    d2.add_line('w5')
    d2.add_line('w6')
    d2.add_line('w7')
    d2.add_line('w8')
    d2.add_line('RMSVE')
    d2.add_line('RMSPBE')

    # problem variables
    n_bin = last_bin
    num_state = n_bin + 1
    active_features = 1
    num_action = 2 # dashed - solid
    alpha_ETD = 0.005/active_features
    alpha_GTD = 0.005/active_features
    beta = 0.05 /active_features
    lam = 0.9

    # time step
    t = 0

    # bin config
    bins = np.linspace(0, 3, n_bin, endpoint=False)


    # previous time step variables
    last_state = None
    initial_w = [1,1,1,1,1,1,10,1]

    # ETD and GTDsetup
    etd_algo = ETD(num_state)
    gtd_algo = GTD(num_state)

    gtd_algo.set_initial_weights(copy.deepcopy(initial_w))
    etd_algo.set_initial_weights(copy.deepcopy(initial_w))

    # RM
    RMSVE_GTD = []
    RMSPBE_GTD = []
    RMSVE_ETD = []
    RMSPBE_ETD = []
    while True:

        # reading data for servo 1
        [ang, position, speed, load, voltage, temperature] = read_data(s1)

        current_state = get_angle_bin(ang,bins)
        action = behaviorPolicy(current_state)
        takeAction(current_state,action,s1)
        # TD lambda
        state = last_state
        state_prime = current_state

        if not last_state == None:

            # verifier
            reward = cummlant(state)
            rho = get_rho(state_prime,action)
            # rho = get_rho(state,action)
            gtd_algo.update(feature_vector(state),reward,feature_vector(state_prime),alpha_GTD,beta,gamma(state),gamma(state_prime),lam,lam,rho)
            etd_algo.update(feature_vector(state),reward,feature_vector(state_prime),alpha_ETD,gamma(state),gamma(state_prime),lam,rho,1)

        # get the current weights
        gtd_w = copy.deepcopy(gtd_algo.get_weights())
        etd_w = copy.deepcopy(etd_algo.get_weights())

        # update the RM
        RMSVE_GTD.append(computeRMSVE(copy.deepcopy(gtd_w)))
        RMSPBE_GTD.append(computeRMSPBE(copy.deepcopy(gtd_w)))

        RMSVE_ETD.append(computeRMSVE(copy.deepcopy(etd_w)))
        RMSPBE_ETD.append(computeRMSPBE(copy.deepcopy(etd_w)))

        # plot and save data
        d1.update(t,[gtd_w[0],gtd_w[1],gtd_w[2],gtd_w[3],gtd_w[4],gtd_w[5],gtd_w[6],gtd_w[7], RMSVE_GTD[t],RMSPBE_GTD[t]])
        d2.update(t,[etd_w[0],etd_w[1],etd_w[2],etd_w[3],etd_w[4],etd_w[5],etd_w[6],etd_w[7], RMSVE_ETD[t],RMSPBE_ETD[t]])
        gtd_data.append([t,gtd_w[0],gtd_w[1],gtd_w[2],gtd_w[3],gtd_w[4],gtd_w[5],gtd_w[6],gtd_w[7], RMSVE_GTD[t],RMSPBE_GTD[t]])
        etd_data.append([t,etd_w[0],etd_w[1],etd_w[2],etd_w[3],etd_w[4],etd_w[5],etd_w[6],etd_w[7], RMSVE_ETD[t],RMSPBE_ETD[t]])
        # go to the next time step
        t += 1
        last_state = current_state
        if flag_stop:
            thread.exit_thread()


# write plotting data to file before ending by ctrl+c
def signal_handler(signal, frame):
    global flag_stop, servo1_data, gtd_data, etd_data

    # stop threads
    flag_stop = True

    # now we need to dump the sensorimotor datastream to disk
    np_servo1_data = np.asarray(servo1_data)
    np_gtd_data = np.asarray(gtd_data)
    np_etd_data = np.asarray(etd_data)
    #np.savetxt('baird_data.txt', np_servo1_data)
    np.savetxt('etd_data.txt', np_etd_data)
    np.savetxt('gtd_data.txt', np_gtd_data)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()

