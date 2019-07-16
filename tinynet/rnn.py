import numpy as np
from ffnn import FFNN

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
        net = RNN([2,2,1])
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

