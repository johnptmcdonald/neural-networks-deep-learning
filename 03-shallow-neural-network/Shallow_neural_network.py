
"""
A neural network is essentially the same calculations that were done in logistic regression, just repeated many times.

A quick word on notation:

Because we now have multiple neurons in multiple layers, the notation must be very specific. We use superscript to identify the neuron layer, and subscript to identify the neuron. 


So our weights matrix will now be a matrix of column vectors, where each column represents the weights for a single neuron in the first layer:

w = [
	[0.4, 1.6],
	[0.2, 1.7],
	[0.1, 1.6],
	[0.7, 1.9],
]

i.e. the first neuron in the first layer has weights of...

	[
		[0.4],
		[0.2],
		[0.1],
		[0.7]
	]

...and the second neuron in the first layer has weights of....


	[
		[1.6],
		[1.7],
		[1.6],
		[1.9]
	]


b is now a column vector of biases:

b = [
	[1], // bias for first neuron
	[2]	 // bias for second neuron
]

Thus our raw activation is still np.dot(w.T, X) + b, which results in column vectors of raw activations for each neuron, which then has the b column vector added to it, which can then be sigmoided. 


In general, different nodes in a layer are stacked vertically. 

"""













