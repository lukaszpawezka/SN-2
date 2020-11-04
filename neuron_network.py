import numpy as np


def get_output_array(result, size):
    arr = np.zeros((size, 1))
    arr[result] = 1
    return arr


def cost_function(a, y):
    return a - y


class NeuronNetwork:
    def __init__(self,
                 layers,
                 init_range_scale,
                 activation_function,
                 momentum=0,
                 negative=True,
                 soft_max_output=False
                 ):
        self.soft_max_output = soft_max_output
        self.function, self.prime = activation_function
        self.layers_details = layers
        self.momentum = momentum
        self.init_range_scale = init_range_scale
        self.z_values = []
        self.activations = []
        self.old_err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        self.old_err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
        if negative:
            self.wights = [(np.random.random((layer[1], layer[0])) - 0.5) * 2 * self.init_range_scale for layer in
                           self.layers_details]
            self.biases = [(np.random.random((layer[1], 1)) - 0.5) * 2 * self.init_range_scale for layer in
                           self.layers_details]
        else:
            self.wights = [(np.random.random((layer[1], layer[0]))) * self.init_range_scale for layer in
                           self.layers_details]
            self.biases = [(np.random.random((layer[1], 1))) * self.init_range_scale for layer in
                           self.layers_details]

    def forward_for_learning_fast(self, input):
        self.activations = [input.reshape((-1, 1))]
        self.z_values = []
        for w, b in zip(self.wights, self.biases):
            self.z_values.append(w @ self.activations[-1] + b)
            self.activations.append(self.function(self.z_values[-1]))

        # softmax
        if self.soft_max_output:
            exp_values = np.exp(self.activations[-1])
            output = exp_values / np.sum(exp_values, axis=0, keepdims=True)
            self.activations[-1] = output
        return self.activations[-1]

    def forward_for_testing(self, input):
        activation = input.reshape((-1, 1))

        for w, b in zip(self.wights, self.biases):
            z_value = w @ activation + b
            activation = self.function(z_value)

        # softmax
        if self.soft_max_output:
            exp_values = np.exp(activation)
            output = exp_values / np.sum(exp_values, axis=0, keepdims=True)
            return output
        return activation

    def backpropagate(self, input, result):
        self.forward_for_learning_fast(input)

        y = get_output_array(result, self.layers_details[-1][1])
        cost = cost_function(self.activations[-1], y)

        delta = cost * self.prime(self.z_values[-1])
        err_b = [delta]
        err_w = [delta @ self.activations[-2].T]

        for layer_index in range(2, len(self.wights) + 1):
            z_value = self.z_values[-layer_index]
            prime = self.prime(z_value)
            delta = (self.wights[-layer_index + 1].T @ delta) * prime
            err_b.append(delta)
            err_w.append(delta @ self.activations[-1 - layer_index].T)
        err_b.reverse()
        err_w.reverse()
        return err_b, err_w

    def _train_mini_batch(self, mini_batch, mini_batch_results, eta):
        err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
        for input_example, result in zip(mini_batch, mini_batch_results):
            err_b_new, err_w_new = self.backpropagate(input_example, result)
            err_b = [ob + nb for ob, nb in zip(err_b, err_b_new)]
            err_w = [ow + nw for ow, nw in zip(err_w, err_w_new)]

        size = len(mini_batch)
        err_b = [eta * b / size + ob * self.momentum for b, ob in zip(err_b, self.old_err_b)]
        err_w = [eta * w / size + ow * self.momentum for w, ow in zip(err_w, self.old_err_w)]

        self.wights = [w - nw for w, nw in zip(self.wights, err_w)]
        self.biases = [b - nb for b, nb in zip(self.biases, err_b)]

        self.old_err_w = err_w
        self.old_err_b = err_b

    def train(self, input, values, epochs, mini_batch_size, eta):
        for epoch in range(epochs):
            print("Epoch: ", (epoch + 1), "/", epochs)
            state = np.random.get_state()
            np.random.shuffle(input)
            np.random.set_state(state)
            np.random.shuffle(values)

            mini_batches = [input[b:b + mini_batch_size] for b in range(0, int(np.shape(input)[0] / mini_batch_size))]
            mini_batches_results = [values[b:b + mini_batch_size] for b in
                                    range(0, int(np.shape(values)[0] / mini_batch_size))]
            num = 0
            size = len(mini_batches)
            for batch, result in zip(mini_batches, mini_batches_results):
                self._train_mini_batch(batch, result, eta)
                self.old_err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
                self.old_err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
                num += 1
                print(f"{num / size * 100}%")

        pass

    def test(self, input, values):
        print("Testowanie")
        positive = 0
        iter = 0
        for example, value in zip(input, values):

            result = np.argmax(self.forward_for_testing(example))
            if result == value:
                positive += 1
            iter += 1
        print(f"{positive}/{np.shape(values)[0]} = {positive / np.shape(values)[0] * 100}%")
        return positive / np.shape(values)[0] * 100

    def predict(self, input):
        return np.argmax(self.forward_for_testing(input))
