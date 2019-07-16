import unittest
import numpy as np
from tinynet import FFNN, RNN

# TODO: write actual assertions :) just fix a seed and compare results
class TestNet(unittest.TestCase):

    def helper(self, net_class):

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
            net = net_class([2,2,1])
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

    def test_ffnn(self):
        print("\nTesting FFNN")
        self.helper(FFNN)

    def test_rnn(self):
        print("\nTesting RNN")
        self.helper(RNN)


if __name__ == '__main__':
    unittest.main()
