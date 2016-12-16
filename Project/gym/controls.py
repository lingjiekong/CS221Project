import numpy as np
import lunarLanderApi
from lunarLanderApi import Rocket
from copy import deepcopy

# possible actions
g_noEngineFired = 0
g_leftEngineFired = 1
g_bottomEngineFired = 2
g_rightEngineFired = 3

# state list
# 0 : horizontal coord, 1: vert. coord, 2: hor. speed, 3: vert. speed
# 4: angle, 5: angSpeed, 6: legLeft, 7: legRight
horPosWeight = 0.5
horSpeedWeight = 1.
angleThresh = 0.4
targetVertWeight = 0.2
angleMax = 0.05

# controller gain
Kp_Vert = 0.5 # propotional gain for vertical position
Kd_Vert = 1 # derivative gain for vertical velocity
Kp_ang = 0.5 # propotional gain for angular orientation
Kd_ang = 1 # derivative gain for angular velocity


#######################################################################
# algorithm: controller heuristic
def controlsCalc(environ, state):
    # get state information
    horPos = state[0]
    verPos = state[1]
    horVel = state[2]
    verVel = state[3]
    angOri = state[4]
    angVel = state[5]
    lfLeg = state[6]
    RgLef = state[7]

    # target angle: should be limited by horizontal speed and position 
    targetAngle = horVel*horSpeedWeight + horPos*horPosWeight
    if targetAngle < -angleThresh: 
        targetAngle = -angleThresh
    elif targetAngle >  angleThresh: 
        targetAngle =  angleThresh
    
    # target vertical position: should be 0.2 of current position
    targetVertPos = targetVertWeight*np.abs(verPos) 

    # PD controller for angle and vertical command
    newAngle = (targetAngle - angOri)*Kp_ang + (0 - angVel)*Kd_ang
    newVert = (targetVertPos - verPos)*Kp_Vert + (0 - verVel)*Kd_Vert

    # leg contact the ground
    if lfLeg or RgLef:
        newAngle = 0
        newVert = -(verVel)*Kd_Vert

    # band band control
    action = g_noEngineFired
    if newAngle > angleMax: 
        action = g_leftEngineFired
    elif newVert > angleMax and newVert > np.abs(newAngle): 
        action = g_bottomEngineFired
    elif newAngle < -angleMax: 
        action = g_rightEngineFired

    return action


#######################################################################
# model
if __name__=="__main__":
    iter = 0
    totalRewardsCombined = 0
    while iter != 400:
        environ = Rocket()
        state = environ.reset()
        total_reward = 0
        time_steps = 0

        while True:
            action = controlsCalc(environ, state)
            state, reward, gameOver, not_needed = environ.step(action)
            environ.render()
            total_reward += reward
            if gameOver:
                print(total_reward)
                break
            time_steps += 1

        iter += 1
        totalRewardsCombined += deepcopy(total_reward)
    print(totalRewardsCombined/400)
