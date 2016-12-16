
import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import seeding

FPS = 50
game_speed  = float(30)
g_random = 1500.

# Given bounds from problem
g_maxBottomEngine  = 13.0
g_sideEngine  =  0.6
g_rocketBounds =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
g_lengthOfSideLegs = 20
g_lengthBottomLegs = 18
g_legWidth = 2
g_legHeight = 8
g_legSprTorque = 40

g_heightSideEngine = 14.0
g_sideEngineSize   = 12.0

g_winWidth = 600
g_winHeight = 400
g_numLegs = 2

# Used to check if legs touch ground
class CheckLanding(contactListener):
    def __init__(self, environ):
        contactListener.__init__(self)
        self.environ = environ
    def BeginContact(self, touchGround):
        if self.environ.lander==touchGround.fixtureA.body or self.environ.lander==touchGround.fixtureB.body:
            self.environ.game_over = True
        for i in range(g_numLegs):
            if self.environ.legs[i] in [touchGround.fixtureA.body, touchGround.fixtureB.body]:
                self.environ.legs[i].ground_contact = True
    def EndContact(self, touchGround):
        for i in range(g_numLegs):
            if self.environ.legs[i] in [touchGround.fixtureA.body, touchGround.fixtureB.body]:
                self.environ.legs[i].ground_contact = False


class Rocket(gym.Env):
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

        high = np.array([1]*8) 
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(8)

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
        self.world.contactListener_keepref = CheckLanding(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = g_winWidth/game_speed
        H = g_winHeight/game_speed

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

        initial_y = g_winHeight/game_speed
        self.lander = self.world.CreateDynamicBody(
            position = (g_winWidth/game_speed/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/game_speed,y/game_speed) for x,y in g_rocketBounds ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-g_random, g_random),
            self.np_random.uniform(-g_random, g_random)
            ), True)

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (g_winWidth/game_speed/2 - i*g_lengthOfSideLegs/game_speed, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(g_legWidth/game_speed, g_legHeight/game_speed)),
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
                localAnchorB=(i*g_lengthOfSideLegs/game_speed, g_lengthBottomLegs/game_speed),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=g_legSprTorque,
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

        return self._step(np.array([0,0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x,y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/game_speed, pos=(0,0)),
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
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / game_speed for _ in range(2)]

        m_power = 0.0
        if not self.continuous and action==2:
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/game_speed + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/game_speed + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*g_maxBottomEngine*m_power,  oy*g_maxBottomEngine*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*g_maxBottomEngine*m_power, -oy*g_maxBottomEngine*m_power), impulse_pos, True)

        s_power = 0.0
        if not self.continuous and action in [1,3]:
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*g_sideEngineSize/game_speed)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*g_sideEngineSize/game_speed)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/game_speed, self.lander.position[1] + oy + tip[1]*g_heightSideEngine/game_speed)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*g_sideEngine*s_power,  oy*g_sideEngine*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*g_sideEngine*s_power, -oy*g_sideEngine*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - g_winWidth/game_speed/2) / (g_winWidth/game_speed/2),
            (pos.y - (self.helipad_y+g_lengthBottomLegs/game_speed)) / (g_winWidth/game_speed/2),
            vel.x*(g_winWidth/game_speed/2)/FPS,
            vel.y*(g_winHeight/game_speed/2)/FPS,
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
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]   # And ten points for legs touchGround, the idea is if you
                                                              # lose touchGround again after landing, you get negative reward
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
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(g_winWidth, g_winHeight)
            self.viewer.set_bounds(0, g_winWidth/game_speed, 0, g_winHeight/game_speed)

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
            flagy2 = flagy1 + 50/game_speed
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/game_speed), (x+25/game_speed, flagy2-5/game_speed)], color=(0.8,0.8,0) )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')