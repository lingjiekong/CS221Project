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
newVertWeight = 0.5
newAngleWeight = 0.5
newAngularSpeedWeight = 1.
def controlsCalc(environ, state):
    # angle should be limited by horizontal speed and position
    targetAngle = state[0]*horPosWeight + state[2]*horSpeedWeight
    if targetAngle >  angleThresh: 
        targetAngle =  angleThresh
    if targetAngle < -angleThresh: 
        targetAngle = -angleThresh

    targetVertPos = np.abs(state[1])*targetVertWeight

    newAngle = (targetAngle - state[4])*newAngleWeight - (state[5])*newAngularSpeedWeight
    newVert = (targetVertPos - state[1])*newVertWeight - (state[3])*newVertWeight

    if state[6] or state[7]:
        newAngle = 0
        newVert = -(state[3])*newVertWeight

    action = g_noEngineFired
    if newAngle > angleMax: 
        action = g_leftEngineFired
    elif newVert > np.abs(newAngle) and newVert > 0.05: 
        action = g_bottomEngineFired
    elif newAngle < -angleMax: 
        action = g_rightEngineFired

    return action

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
