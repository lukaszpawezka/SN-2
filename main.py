from neuron_network import NeuronNetwork
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import random
f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()

tt, vt = training_data
tv, vv = validation_data
# print(np.shape(t), np.shape(v))

# print(training_data)
# print(np.shape(validation_data))
# print(np.shape(test_data))

layers = [
    (784, 60),
    (60, 10)
]


def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return 1. * (x > 0)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

relu_activation_function = {
    lambda x: relu(x),
    lambda x: relu_derivative(x)
}

sigmoid_activation_function = (
    lambda x: sigmoid(x),
    lambda x: sigmoid_prime(x)
)

net_relu = NeuronNetwork(
    layers,
    0.1,
    relu_activation_function,
    momentum=0.3,
    negative=True,
    soft_max_output=True
)

net_sigmoid = NeuronNetwork(
    layers,
    0.5,
    sigmoid_activation_function,
    momentum=0.1,
    negative=True,
    soft_max_output=True
)

# def singular_run():
#
#     print(np.shape(tt), np.shape(vt))
#
#     start = time.time()
#     nn.train(tt, vt, 3, 200, 0.05)
#     print(round((time.time()-start)*100)/100, "s")
#
#     start = time.time()
#     nn.test(tv, vv)
#     print(round((time.time()-start)*100)/100, "s")
#
#     index = random.randint(0, len(tv))
#     print("Przewidywania dla: ", nn.predict(tv[index]))
#
#     plt.imshow(np.resize(tv[index], (28, 28)))
#     plt.show()
# singular_run()

def plot_run():
#    data_for_plot = []
    data_for_plot2 = []
    start_relu = time.time()
    for i in range(0, 10):
#        net_relu.train(tt, vt, 1, 200, 0.1)
#        data_for_plot.append(net_relu.test(tv, vv))
        net_sigmoid.train(tt, vt, 1, 100, 100)
        data_for_plot2.append(net_sigmoid.test(tv, vv))
    end_relu = 'Czas relu: ' + str(round((time.time() - start_relu) * 100) / 100) + ' s'
    print(end_relu)
    plt.annotate(str(round(data_for_plot2[i], 2)) + '%', (i, data_for_plot2[i]-5))
#    plt.plot(data_for_plot, 'b', marker='o', label='Relu')
    plt.plot(data_for_plot2, 'b', marker='o', label='Sigmoid')
#    plt.legend(loc='lower right')
    plt.suptitle('Wykres procentowego stanu wyuczenia modelu w danej epoce')
    plt.xlabel("Numer epoki")
    plt.ylabel("Procent wyuczenia modelu")
    plt.xticks(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
               ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.savefig('wykres1')
    plt.show()

plot_run()

# print(np.array([1,4,1,2,5,6])*np.array([2,1,4,6])[:, np.newaxis])
#
#
# inp = np.array([
#     [1, 0, 0],
#     [0, 1, 1],
# ])
#
# neur = np.array([
#     [10, 2, 2, 2],
#     [10, 3, 3, 3]
# ])
#

# print(np.tile(inp[0], (neur.shape[0], 1)))
# print(neur[:, 0])
# print(np.sum(np.tile(inp[0], (neur.shape[0], 1))*neur[:, 1:], axis=1)+neur[:, 0])
# print(np.sum(inp[0]*neur[:, 1:], axis=1)+neur[:, 0])

# delta = np.array([
#     1,2,3,4,5
# ])
#
# prime = np.array([
#     0,0,1,0,0,1,0
# ])
#
# layers = np.array([
#     [1, 0, 0, 0, 0],
#     [1, 1, 0, 0, 0],
#     [1, 0, 1, 0, 0],
#     [1, 0, 0, 1, 0],
#     [1, 0, 0, 0, 1],
#     [1, 0, 0, 1, 0],
#     [1, 0, 1, 0, 0],
# ])


#
# print(
#     np.array([
#         [1,1,2,3,5,2,3,5,2],
#         [1,3,2,3,4,6,2,1,2],
#         [2,4,2,5,6,2,6,7,2]
#     ]).T *
#     np.array([1,0,1])
# )
#
#
# print(
#     np.sum(
#     np.array([
#         [1,1,2,3,5,2,3,5,2],
#         [1,3,2,3,4,6,2,1,2],
#         [2,4,2,5,6,2,6,7,2]
#     ]).T *
#     np.array([1,0,1]), axis=1)
# )

# print(layers*prime[:, np.newaxis])
# print(layers*delta)
#
# print(np.array([1,0,0,0])*np.array([2,2,2,2]))
#
# print(np.shape(layers), np.shape(delta))
# print(np.sum(layers*delta, axis=1))