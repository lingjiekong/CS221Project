import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

import collections # use for sparse vector dot product
import random # use for generate random value initally

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# Too see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v0
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class LunarLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):


        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        high = np.array([np.inf]*8)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,) )
        chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y  = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i],   smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append( [p1, p2, (p2[0],H), (p1[0],H)] )

        self.moon.color1 = (0.0,0.0,0.0)
        self.moon.color2 = (0.0,0.0,0.0)

        initial_y = VIEWPORT_H/SCALE
        self.lander = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        # print ('from reset module')
        # print (self._step(np.array([0,0]) if self.continuous else 0)[0])

        return self._step(np.array([0,0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x,y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl<0):
            self.world.DestroyBody(self.particles.pop(0))

    def _step(self, action):
        # print ('action')
        # print (action)
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0): # or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*MAIN_ENGINE_POWER*m_power,  oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5): # or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        assert len(state)==8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]   # And ten points for legs contact, the idea is if you
                                                              # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done   = True
            reward = -100
        if not self.lander.awake:
            done   = True
            reward = +100
        # print ('in step module')
        # print ('state')
        # print (state)
        # print ('np.array(state)')
        # print (np.array(state))
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0,0,0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    #########################################################################
    # start of my code and above the lunarlander API
    def getState(self, s):
        # s is the state information in which: 
        # s[0] is horizontal coordiante
        # s[1] is vertical coordinate
        # s[2] is horizontal speed
        # s[3] is vertical speed
        # s[4] is angle
        # s[5] is angular speed
        # s[6] is right leg contact (1) and not contact is 0
        # s[7] is left leg contact (1) and not contact is 0
        state = 0
        # 3 x / 3 xdot / 3 y / 3 ydot / 3 theta / 3 thetadot / 1 succeed / 1 fail and 0 is count as a state
        totalState = 3*3*3*3*3*3+1 
        # mapping the state information
        if (self.game_over or abs(s[0]) >= 1.0):
            state = totalState  # fail the landing 730
        elif not self.lander.awake:
            state = totalState - 1 # succeed the landing 729
        else:
            # s[0]
            if (s[0] < -0.1):
                state = 0
            elif (s[0] < 0.1):
                state = 1
            else:
                state = 2

            # s[1]
            if (s[1] < 0.15):
                state += 0
            elif (s[1] < 0.5):
                state += 3
            else:
                state += 6

            # s[2]
            if (s[2] < -0.2):
                state += 0
            elif (s[2] < 0.2):
                state += 9
            else:
                state += 18

            # s[3]
            if (s[3] < -0.2):
                state += 0
            elif (s[3] < 0.2):
                state += 27
            else:
                state += 54

            # s[4]
            if (s[4] < -0.1):
                state += 0
            elif (s[4] < 0.1):
                state += 81
            else:
                state += 162

            # s[5]
            if (s[5] < -0.1):
                state += 0
            elif (s[5] < 0.1):
                state += 243
            else:
                state += 486
        # print (state)
        return state # value between 0 and 730


class LunarLanderContinuous(LunarLander):
    continuous = True

if __name__=="__main__":
    #########################################################################
    # define convergence threshold
    NUM_STATES = 3*3*3*3*3*3+1
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20
    LEFT_RIGHT_ACTION = 3
    BOTTOM_ACTION = 2

    #########################################################################
    # define MPD traning parameters
    # count time for transition
    tranCount = collections.defaultdict(float) 

    # real transition probability 
    tranProb = collections.defaultdict(float)
    for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION): # 29 different actions include 0
        tranProb[a] = collections.defaultdict(int)
        for startState in range(0, NUM_STATES + 1):
            tranProb[a][startState] = collections.defaultdict(float)
            for endState in range(0, NUM_STATES + 1):
                tranProb[a][startState][endState] = 1.0/float(NUM_STATES)

    # count reward time + 1
    rewardCountTime = collections.defaultdict(float)
    # count reward value + R
    rewardCountValue = collections.defaultdict(float)
    # real reward used in calcualtion
    reward = collections.defaultdict(float)
    # real value used in calcualtion and value iteration 
    value = collections.defaultdict(float)
    for s in range(0,NUM_STATES+1):
        value[s] = 0.1*random.random()
    value[729] = 100
    #########################################################################
    # define the action mapping
    actionDict = collections.defaultdict(float)
    for r in range(0,BOTTOM_ACTION): # BOTTOM_ACTION = 2
        for lr in range(0,LEFT_RIGHT_ACTION): # LEFT_RIGHT_ACTION = 3
            actionDict[r*3+lr] = [float(r), float(lr)-1]

    #########################################################################
    # get initial state information of game
    env = LunarLanderContinuous()
    state = env.reset()

    #########################################################################
    # learning
    consecutiveNoLearningTrials = 0
    trainingCount = 0
    updateCount = 0
    while(trainingCount < 400):
    # while(consecutiveNoLearningTrials < NO_LEARNING_THRESHOLD):
        # find what is the best action to take
        maxActionList = list()
        maxActionValue = -float('inf')
        actionValue = 0.0
        stateMap = env.getState(state)
        for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION):
            for i in list(tranProb[a][stateMap]):
                # tranProb and value is the same every iteration
                actionValue += tranProb[a][stateMap][i] * value[i]
            if actionValue > maxActionValue:
                maxActionValue = actionValue
                maxActionList = list()
                maxActionList.append(a)
            elif actionValue == maxActionValue:
                maxActionList.append(a)
            actionValue = 0
        maxAction = random.choice(maxActionList)

        # get real action from the mapping 
        realAction = np.array(actionDict[maxAction])

        # get the newS/reward/done by simulating the dynamics
        newState, r, done, info = env.step(realAction)
        # env.render()
        newStateMap = env.getState(newState)

        # count for transition probablity 
        if tranCount[maxAction] == 0: 
            tranCount[maxAction] = collections.defaultdict(float) 
            tranCount[maxAction][stateMap] = collections.defaultdict(float)
            tranCount[maxAction][stateMap][newStateMap] = 1    
        elif tranCount[maxAction][stateMap] == 0: 
            tranCount[maxAction][stateMap] = collections.defaultdict(float)
            tranCount[maxAction][stateMap][newStateMap] = 1    
        else: # 
            tranCount[maxAction][stateMap][newStateMap] += 1

        # count for reward 
        rewardCountTime[newStateMap] += 1
        if (newStateMap == NUM_STATES or newStateMap == NUM_STATES - 1):
            rewardCountValue[newStateMap] += (r + (-abs(newState[0])-abs(newState[2])-abs(newState[3])-abs(newState[4])-abs(newState[5])))
        else:
            rewardCountValue[newStateMap] += (-abs(newState[0])-abs(newState[2])-abs(newState[3])-abs(newState[4])-abs(newState[5]))

        # End of one episole of game(win\lose) use value iteration to update policy
        if (newStateMap == NUM_STATES or newStateMap == NUM_STATES - 1):
            # update reward
            for i in list(rewardCountTime):
                reward[i] = float(rewardCountValue[i])/float(rewardCountTime[i])

            # update transition probability
            for a in list(tranCount): # each action
                for i in list(tranCount[a]): # each start state
                    totalCount = sum((tranCount[a][i]).values()) # sum total count in start -> end
                    DummyTranProb = collections.defaultdict(float)
                    for j in list(tranCount[a][i]):
                        DummyTranProb[j] = float(tranCount[a][i][j])/float(totalCount)
                    tranProb[a][i] = DummyTranProb

            # value iteration
            updateValue = collections.defaultdict(float)
            updateCount = 0
            vOpt = 0
            err = float('inf')
            while err > TOLERANCE:
                for i in range(0, NUM_STATES + 1):
                    vOptMax = -float('inf')
                    for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION):
                        for j in list(tranProb[a][i]):
                            vOpt += tranProb[a][i][j]*value[j]
                        if vOpt > vOptMax:
                            vOptMax = vOpt
                        vOpt = 0
                    updateValue[i] = reward[i]+GAMMA*vOptMax
                updateCount += 1
                err = max(abs(updateValue[i] - value[i]) for i in list(value))
                value = updateValue

            # check convergence to stop training
            if updateCount == 1:
                consecutiveNoLearningTrials += 1
            else:
                consecutiveNoLearningTrials = 0
            trainingCount += 1
            print (reward[newStateMap])
            env = LunarLanderContinuous()
            state = env.reset()

    #########################################################################  
    # Start play the game with the learned T, R, and V
    env = LunarLanderContinuous()
    # s is the state of the game
    s = env.reset()
    total_reward = 0
    steps = 0
    iteration = 10;
    # use MDP to play the game
    while True:
        for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION):
            for i in list(tranProb[a][stateMap]):
                actionValue += tranProb[a][stateMap][i] * value[i]
            if actionValue > maxActionValue:
                maxAction = a
            actionValue = 0

        # get real action from the mapping 
        realAction = np.array(actionDict[maxAction])
        # env.step(a) implement action a and render the environment
        s, r, done, info = env.step(realAction)
        env.render()
        total_reward += r
        steps += 1
        if done: break
    print ('value')
    print (value)
    print ('reward')
    print (reward)
    dummyCount = 0
    for a in range(0,LEFT_RIGHT_ACTION*BOTTOM_ACTION): # 29 different actions include 0
        for startState in range(0, NUM_STATES + 1):
            for endState in range(0, NUM_STATES + 1):
                if tranProb[a][startState][endState] !=  0.0013698630136986301:
                    dummyCount += 1
    print ('dummyCount')
    print (dummyCount)
