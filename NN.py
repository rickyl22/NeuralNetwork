from random import random
from math import exp
import matplotlib.pyplot as plt

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, file_out):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		# e = str(sum_error) + "\n"
		# file_out.write(e)
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def read_data(filename):
	data = []
	f = open(filename, 'r')
	for line in f:
		row = []
		aux = line.split()
		for i in aux:
			row.append(float(i))
		row[-1] = int(row[-1])
		data.append(row)
	return data


n_inputs = 2
n_outputs = 2
neurons = 20

           
network = initialize_network(n_inputs, neurons, n_outputs)
data = read_data("datos_P2_EM2017_N2000.txt")
errores = open("errores_2000_10.txt",'w')
colores = open("colores_2000_10.txt", "w")
test = open("test.txt", "w")
for i in range(2000):
     x = random()*20
     y = random()*20
     if (x<=7.5 or x >= 12.5) and (y<=7.5 or y>=12.5):
             test.write("%.3f  %.3f  %d\n" % (x,y,0))
     else:
             test.write("%.3f  %.3f  %d\n" % (x,y,1))
test = read_data("test.txt")
train_network(network, test, 0.1, 1000, n_outputs, errores)
prueba = read_data("prueba.txt")
samp = read_data("sample.txt")
for row in samp:
    outputs = forward_propagate(network, row)
    x = str(row[0])
    y = str(row[1])
    c = str(outputs.index(max(outputs)))
    l = x + " " + y + " " + c + "\n"
    colores.write(l)
    #print("%.3f  %.3f  %d" % (row[0],row[1],outputs.index(max(outputs))))

result = read_data("colores_2000_10.txt");
for line in result:
        if(line[2] == 1):
                plt.plot(line[0],line[1], 'b.')
        else:
                plt.plot(line[0],line[1], 'r.')

plt.show()

