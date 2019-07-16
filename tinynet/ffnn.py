import numpy as np

class FFNN:
    """Fully connected feed-forward multilayer perceptron (neural network)

    - `self.state`: list of inputs to each layer (+ final output)
    - `self.weights`: list of weight matrices
    - rows in a weight matrix correspond to weights of connections entering a same neuron
    - cols in a weight matrix correspond to connections from the same input
    """

    def __init__(self, struct, act_fn=np.tanh, init_weights=None):
        assert len(struct) >= 2, "`struct` needs at least input and output size"
        # TODO: assert act_fn applies element-wise to a np array
        self.act_fn = act_fn
        # Compute sizes and shapes from net struct
        self.struct = struct
        self.ninputs, *self.nhid, self.noutputs = self.struct # user convenience
        self.nlayers = len(self.struct) - 1 # first number in inputs, not a layer
        self.act_sizes = self.struct[1:]
        # State index accessors
        self.input_idxs = [range(0, ninputs) for ninputs in self.struct[:-1]]
        self.bias_idxs = [range(ninputs, ninputs + 1) for ninputs in self.struct[:-1]]
        self.act_idxs = self.input_idxs[1:] + [range(0, self.struct[-1])] # last state is all act
        # Compute size of inputs to each layer
        self.input_sizes = [nins + 1 for nins in self.struct[:-1]]
        # Compute all dimensions depending on layer input sizes
        self._reset_sizes() # => moved to a function for easier call in subclasses (check RNN)
        # Fill weight matrices
        self.set_weights(init_weights or np.random.randn(self.nweights))

    def _reset_sizes(self):
        self.state_sizes = self.input_sizes + [self.struct[-1]] # last state is only output
        self.wmat_shapes = list(zip(self.act_sizes, self.input_sizes))
        self.wmat_sizes = [nneurs * ninputs for nneurs, ninputs in self.wmat_shapes]
        self.nweights_per_layer = self.wmat_sizes # user convenience
        self.nweights = sum(self.wmat_sizes)

    def reset_state(self):
        if hasattr(self, 'state'): del self.state # preemptive GC call
        self.state = []
        for state_size, bias_idx in zip(self.state_sizes, self.bias_idxs):
            layer_state = np.zeros(state_size)
            layer_state[bias_idx] = 1
            self.state.append(layer_state)
        self.state.append(np.zeros(self.state_sizes[-1])) # last state is all act

    def set_weights(self, weights):
        assert weights.size == self.nweights, "Wrong number of weights"
        if hasattr(self, 'weights'): del self.weights # preemptive GC call
        self.weights = []
        # Partition the weights per each matrix (but seriously, no numpy fn for this??)
        idx = 0
        for size, shape in zip(self.wmat_sizes, self.wmat_shapes):
            self.weights.append(weights[idx:idx+size].reshape(shape))
            idx += size
        self.reset_state()

    def activate(self, inputs):
        # Set input to first layer
        self.state[0][self.input_idxs[0]] = inputs
        # Activate each layer in turn
        for layer_idx in range(0, self.nlayers):
            self.activate_layer(layer_idx)
        # Return final activation
        return self.get_act()

    def activate_layer(self, layer_idx):
        net = np.dot(self.weights[layer_idx], self.state[layer_idx])
        act = self.act_fn(net)
        # set output into next state (= layer_idx+1)
        self.state[layer_idx+1][self.act_idxs[layer_idx]] = act

    def get_weights(self):
        """Return a single, flattened numpy array with all the network's weights"""
        return np.concatenate([wmat.flatten() for wmat in self.weights])

    def get_act(self):
        return self.state[-1] # last state is only act



################################################################################



if __name__ == "__main__":

    # Print seeds <100 where the task is solved in <1k trials
    max_ntrials = 1000
    nseeds = 100
    successful = 0
    ntrials_acc = 0
    for rseed in range(nseeds):
        np.random.seed(rseed)

        # Classic XOR separation task using network with 2 hid and 1 out
        # NOTE: a FFNN can learn the mapping, but a RNN will learn the sequence
        # The result is that we expect A LOT more seeds to succeed
        net = FFNN([2,2,1])
        xor = np.array([[[0,0],0], [[0,1],1], [[1,0],1], [[1,1,],0]])

        # Neuroevolution fitness function
        def fit(weights):
            net.set_weights(weights)
            error = 0
            for inputs, target in xor:
                output = net.activate(inputs)[0]
                # Continuous error (easier for gradient-based training):
                # ans = output
                # Discrete error (easier for RWG):
                if output<0.5: ans = 0
                else:          ans = 1
                error += abs(ans - target) # (output - target)**2
            return error

        weights = np.random.randn(net.nweights)
        error = 1e10
        # print("\nErrors:")

        # Try guessing the weights 1k times
        for ntrial in range(max_ntrials):
            # Weight generated using random weight guessing (RWG)
            weights = np.random.randn(net.nweights)
            error = fit(weights)
            # print(error, end=' ')
            if error == 0:
                print(f"Seed {rseed}: success in {ntrial} trials")
                successful += 1
                ntrials_acc += ntrial
                break

    print(f"\nSuccessful with {successful}/{nseeds} seeds (avg {round(ntrials_acc/successful,2)} trials).")

    # import IPython; IPython.embed()

