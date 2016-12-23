Author: Stanley Jacob, Lingjie Kong
Created: Dec 16th
Revision: X1


I. File list
------------
1. Control approach
	a. controls.py: solution by using control

	b. lunarLanderAPiControl.py: Api for control approach

2. MDP approach
	a. MDP.py: solution by using MDP

	b. lunarLanderApiMDP.py: APi for MDP approach

3. Q-learing Multilayer Perceptron approach
	a. run.py and agents.py: solution by using Q-learing

	b. unar_lander.py: APi for Q-learing approach


II. Installation
------------
1. Download the OpenAi Gym enviroinment at https://github.com/openai/gym
	a. Install everything: install the full set of the environment by 
		brew install cmake boost boost-python sdl2 swig wget

	b. Supported systems: we currently support Linux and OS X running Python 2.7 or 3.5. Some users on OSX + Python3 may need to run
		brew install boost-python --with-python3

	c. Pip version: To run pip install -e '.[all]', you'll need a semi-recent pip. Please make sure your pip is at least at version 1.5.0
		pip install --ignore-installed pip

	d. Rendering on a server: If you're trying to render video on a server, you'll need to connect a fake display.
		xvfb-run -s "-screen 0 1400x900x24" bash

	e. Environments: The environment for lunarland game is Box2d and it can be installed by
		pip install -e '.[box2d]'

	f. Test of successful installation: for testing the success of installation can be done by:
		import gym
		env = gym.make('LunarLander-v2')
		env.reset()
		env.render()

2. Troubleshoot
	a. If there is any problem on lunching the lunarland Api. This link below will be a good source for troubleshoot.
		https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md


III. Command
------------
1. Running control approach
	python controls.py: is the command to run code for control approach

2. Running MDP approach
	python MPD.py: is the command to run code for MDP approach

3. Running Q-learing approach
	python run.py: is the command to run code for Q-learing approach













