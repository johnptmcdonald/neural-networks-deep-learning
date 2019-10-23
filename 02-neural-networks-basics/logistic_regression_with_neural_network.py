"""
The notebook is located here (requires login to coursera): https://erjsarjplgszrwdfajaeto.coursera-apps.org/notebooks/Week%202/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic_Regression_with_a_Neural_Network_mindset_v6a.ipynb#


0 - Problem Statement: You are given a dataset ("data.h5") containing:

- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).

Number of training examples: m_train = 209
Number of testing examples: m_test = 50
Height/Width of each image: num_px = 64
Each image is of size: (64, 64, 3)
train_set_x shape: (209, 64, 64, 3)
train_set_y shape: (1, 209)
test_set_x shape: (50, 64, 64, 3)
test_set_y shape: (1, 50)


1 - Pre-process the data
------------------------

Common steps for pre-processing a new dataset are:

- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
- "Standardize" the data (in this case we can divide by 255 as each pixel value is the RGB channel value)

In this case we completely flatten everything. We previously had a train_set_x of (209, 64, 64, 3), i.e. 209 images consisting of 3 separate 64 by 64 pixel values (one for each R, G, or B value), and we flatten into (12288, 209) i.e. 12288 rows and 209 columns. 

*As always - the X dataset consists of a number of vertical instances - i.e. each Xi is a standard (vertical) vector of 12288 properties*



2 - Consider the structure of the NN and do a forward pass
----------------------------------------------------------

Given that we have a single neuron in our netowrk, and each instance has 12288 properties, it means our weights (vertical) vector will consist of 12288 values in 1 column, and our bias value with be a scalar. We initialize both with zeros.

We now do a forward pass:

# Calculate (w.T * X) + b, then pass it through s sigmoid function
Activation = σ(np.dot(w.T, X) + b)
		   = horizontal vector corresponding to the activations for each 	instance

# Get total cost by comparing our activations with the correct Y values
cost = -(1/m) * np.sum((Y*np.log(A) + (1-Y)*np.log(1-A)))
	 = scalar (representing goodness of fit)


# Calculate the gradients for each weight and also the bias, wrt to the cost for this batch of X. 

dw = (1/m) * np.dot(X, (A-Y).T)
   = [
		[dw1],
		[dw2],
		[dw3],
		[dw4],
		 ...
		[dwi],		
    ]	
   = vertical vector of all the weights for each of the 12288 properties

We are multiplying the _rows_ of X with (A-Y).T, i.e. summing the dw1 gradients for all the instances, then summing the dw2 gradients for all instances, then summing the dw3 gradients etc. 
Then we divide by 'm' to get the average gradient fo dw1, dw2, dw3, etc.

We do the same for b:

db = (1/m) * np.sum(A-Y)
   = scalar


3 - Optimize the weights and bias/es with gradient descent
----------------------------------------------------------
We then change the weights (and bias) in the opposite direction of the gradients in order to decrease the total cost:

w = w - learning_rate * dw
b = b - learning_rate * db

Then run another forward pass to calculate costs, get the gradients, then optimize weights and bias again.


4 - Use the optimized network to make predictions
-------------------------------------------------
Run an X vector/matrix through the network. If the activation is greater than 0.5, consider it a 1, otherwise consider it a 0.

"""












