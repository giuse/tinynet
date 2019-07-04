# tinynet

A tiny neural network library

## No training

This library provides no training algorithm. Use in conjunction with a
black box search algorithm such as [CMA-ES](https://github.com/CMA-ES/pycma)
to train the weights in a neuroevolution framework.

## Installation

`pip install tinynet`

## Usage

```python
from tinynet import RNN1L
import numpy as np
ninputs, noutputs = [3, 2]
net = RNN1L(ninputs, noutputs)
net.set_weights(np.random.rand(net.nweights()))
out = net.activate(np.zeros(ninputs))
assert len(out) == noutputs
assert len(net.state) == ninputs + 1 + noutputs # input, bias, recursion
```

## Neuroevolution application

```python

import numpy as np
from tinynet import RNN1L
import gym

# Get pre-trained weights
pre_trained_weights = raise "Check out https://gist.github.com/giuse/3d16c947259173d571cf82e28a2f7a7e"

# Environment setup
env = gym.make("BipedalWalker-v2")
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video
nactions = env.action_space.shape[0]
ninputs = env.reset().size

# Network setup
net = RNN1L(ninputs, nactions)
net.set_weights(pre_trained_weights)

# Gameplay loop
obs = env.reset()
score = 0
done = False
while not done:
  env.render()
  action = net.activate(obs)
  obs, rew, done, info = env.step(action)
  score += rew
print(f"Fitness: {score}")
env.close()
```

<!-- 
Why .md instead of .rst? Because I don't want to get such an error ever again:

```bash
$ pipenv run twine check dist/*
Checking distribution dist/tinynet.tar.gz: warning: `long_description_content_type` missing.  defaulting to `text/x-rst`.
Failed
The project's long_description has invalid markup which will not be rendered on PyPI. The following syntax errors were detected:
line 7: Warning: Title underline too short.

No training
----------
```
-->
