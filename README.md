# tinynet

A tiny neural network library

## No training

This library provides no training algorithm. You can easily set up a neuroevolution framework by including any black-box search algorithm, from [RWG](https://www.bioinf.jku.at/publications/older/ch9.pdf) (see example below) to [CMA-ES](https://github.com/CMA-ES/pycma).

## Installation

`pip install tinynet`

## Usage

```python
from tinynet import RNN1L
import numpy as np
net_struct = [3, 5, 2]
net = RNN(net_struct) # try also FFNN
net.set_weights(np.random.randn(net.nweights))
out = net.activate(np.zeros(net.ninputs))
assert len(out) == net.noutputs
assert len(net.state) == net.ninputs + 1 + net.noutputs # input, bias, recursion
```

## Neuroevolution application on the OpenAI Gym

Check out [this GitHub gist](https://gist.github.com/giuse/3d16c947259173d571cf82e28a2f7a7e) to run the Bipedal Walker using pre-trained weights.

The example below tackles the CartPole from scratch using RWG.


```python
import numpy as np
import tinynet
import gym # just `pip install gym`
from time import sleep # slow down rendering

# Environment setup
env = gym.make("CartPole-v1")
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video

# Get input size and output size from the environment
nactions = env.action_space.n
ninputs = env.reset().size
# Hidden layers are arbitrarily added
# hidden = [20, 10, 20]
hidden = [] # ... but unnecessary with the CartPole
net_struct = [ninputs, *hidden, nactions]

# Network setup is straightforward (defaults: `act_fn=np.tanh, init_weights=None`)
net = tinynet.FFNN(net_struct) # also try `RNN(net_struct)`

# Get random seed for deterministic fitness (for simplicity)
rseed = np.random.randint(1e10)

# Fitness function: gameplay loop
def fitness(ind, render=False):
    env.seed(rseed)  # makes fitness deterministic
    obs = env.reset()
    score = 0
    done = False
    net.set_weights(ind)
    while not done:
        if render:
            env.render()
            sleep(0.5)
        action = net.activate(obs).argmax()
        obs, rew, done, info = env.step(action)
        score += rew
    if render: env.render() # render last frame
    print(f"Score: {score}")
    return score

# RWG does not distinguish between populations and generations
max_ninds = 1000

# Neuroevolution (RWG) loop
for nind in range(max_ninds):
    ind = np.random.randn(net.nweights)
    score = fitness(ind)
    if score >= 195:
        print(f"Game solved in {nind} trials")
        break

# Replay winning individual
fitness(ind, render=True)

# You may want to drop into a console here to examine the results
import IPython; IPython.embed()
```
