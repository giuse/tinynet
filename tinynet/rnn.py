import numpy as np
from tinynet.ffnn import FFNN

class RNN(FFNN):
    """Fully connected recurrent multilayer perceptron (neural network)

    - `self.state`: list of inputs to each layer (+ final output)
    - `self.weights`: list of weight matrices
    - rows in a weight matrix correspond to weights of connections entering a same neuron
    - cols in a weight matrix correspond to connections from the same input
    """

    # TODO: *args and **kwargs
    def __init__(self, struct, act_fn=np.tanh, init_weights=None):
        super().__init__(struct, act_fn, init_weights)
        # Compute indices for recurrent inputs
        self.rec_idxs = [range(nin+1, nin+1+nrec) for nin, nrec in zip(self.struct[:-1], self.struct[1:])]
        # Recompute sizes of inputs to each layer
        self.input_sizes = [nin+1+nrec for nin, nrec in zip(self.struct[:-1], self.struct[1:])]
        # Recompute all dimensions depending on layer input sizes
        self._reset_sizes()
        # Fill weight matrices
        self.set_weights(init_weights or np.random.randn(self.nweights))

    def activate_layer(self, layer_idx):
        # Copy act from past iteration into recurrent input
        self.state[layer_idx][self.rec_idxs[layer_idx]] = \
            self.state[layer_idx+1][self.act_idxs[layer_idx]]
        super().activate_layer(layer_idx)
