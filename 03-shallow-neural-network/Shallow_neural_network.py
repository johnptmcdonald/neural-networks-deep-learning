
"""

https://erjsarjplgszrwdfajaeto.coursera-apps.org/notebooks/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar_data_classification_with_onehidden_layer_v6c.ipynb


A neural network is essentially the same calculations that were done in logistic regression, just repeated many times.

A quick word on notation:

Because we now have multiple neurons in multiple layers, the notation must be very specific. We use superscript to identify the neuron layer, and subscript to identify the neuron. 


The notation used in this course is that W1 is equal to the already transposed weight matrix for the first layer.

So if...

X =	[
		[1, 2, 3],
		[1, 2, 3]
	]

i.e. each column represents an instance, each row is a feature

Then the weights for the first layer are repesented as follows:

W1 = [
		[1, 1],		# weights for first neuron in first layer
		[2, 2],		# weights for second neuron in first layer
		[3, 3],		# weights for third neuron in first layer
		[4, 4]		# weights for fourth neuron in first layer
	]

i.e. already transposed, so that... 

W1 * X = [
	[2, 4, 6],
	[4, 8, 12],
	[6, 12, 18],
	[8, 16, 24] 
]

i.e. the first column is the activation for the first instance

A good general rule is that as we look at the neural network, each neuron is its own row in a matrix.



TASK - PLANAR DATA CLASSIFICATION
---------------------------------

We incrementally build a function called nn_model(X, Y, n_h, num_iterations, print_cost=False)

This accepts the X and Y data, the number of hidden layers (n_h), and a couple of other params.

STEP 1 - calculate n_x (input layer size, i.e. number of properties an instance of X has), and n_y (output layer size, i.e. number of output variables - for binary classification this is just 1) from X and Y

STEP 2 - initialize parameters:
Start with random weights for W1 and W2 and zero biases b1 and b2.

STEP 3 - ITERATION
Repeat the following steps many times.
	- STEP 3a: forward_propagation(X, parameters of W1, W2, b1, and b2), which returns output activation A2 and a cache of all intermediate activations Z1, A1, Z2 (to be used later in backprop)
	- STEP 3b: compute_cost(output A2, real Y) to get a measure of error for current parameters
	- STEP 3c: perform backward_propagation(parameters, cache, X, and Y) to get gradients of all weight values at their current values
	- STEP 3d: update parameters by moving in the opposite direction of their gradients

STEP 4 - prediction(parameters, X):
Do a forward propagation to get A2 (output activation), then binarize it to get 0 or 1. 

"""













