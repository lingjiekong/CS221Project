'''
Dec 3, 2016
Stanley Jacob
'''

import gym
import numpy as np
import time
import rocket


if __name__ == '__main__':

    numepisodes = 300                   # Change this to run more training
    seed = 1                            # May not be necessary
    env = gym.make("LunarLander-v2")    # Can be redefined

    print(env.observation_space, env.action_space, env.spec.timestep_limit, env.reward_range, gym.envs.registry.spec("LunarLander-v2").trials)
    params = {
        "memsize": 150000,
        "probupdate": .25,
        "lambda": 0.15,
        "past": 0,
        "eps": 0.75,  # Epsilon in epsilon greedy policies
        "decay": 0.98,  # Epsilon decay in epsilon greedy policies
        "initial_learnrate": 0.012,
        "decay_learnrate": 0.997,
        "discount": 0.99,
        "batch_size": 75,
        "hiddenlayers": [200],
        "regularization": [0.00001, 0.00000001],
    }

    agent = rocket.deepQAgent(env.observation_space, env.action_space, env.reward_range, **params)
    num_steps = env.spec.timestep_limit
    avg = 0.
    oldavg = 0.

    totrewlist = []
    totrewavglist = []
    costlist = []
    showevery = 20
    for episode in range(numepisodes):
        if episode % showevery == 0:
            render = True
            eps = None
            #print('episode', episode, 'l rate', agent.getlearnrate())
            oldavg = avg
        else:
            render = False
            eps = episode
        startt = time.time()
        total_rew, steps, cost, listob, listact = rocket.do_rollout(agent, env, eps, render=render)

        if episode == 0:
            avg = total_rew
        if episode % 1 == 0:
            listob = np.array(listob)
            listact = np.array(listact)
            allactionsparse = np.zeros((listact.shape[0], agent.n_out))
            allactionsparse[np.arange(listact.shape[0]), listact] = 1.

            inc = max(0.06, 1. / (episode + 1.) ** 0.6)
            avg = avg * (1 - inc) + inc * total_rew
            totrewlist.append(total_rew / agent.config['scalereward'])
            totrewavglist.append(avg / agent.config['scalereward'])
            costlist.append(cost / steps)
            plotlast = 200 + episode % 50
        #print('episode', episode, 'l rate', agent.getlearnrate())
        print(total_rew)
        # print(render, 'time', (time.time() - startt) / steps * 100., 'steps', steps, 'total reward', total_rew / \
        #     agent.config['scalereward'], 'avg', avg / agent.config['scalereward'], cost / steps, 'eps',\
        #     agent.epsilon(eps), len(agent.memory))
    
    #print(agent.config)
