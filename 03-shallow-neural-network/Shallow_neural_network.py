
"""
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


TASK - PLANAR DATA CLASSIFICATION
---------------------------------


"""













