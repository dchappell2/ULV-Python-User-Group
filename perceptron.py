# Perceptron Neural Network
# learns to classify output of a binary operator
# inputs and outputs are all binary (0,1) 
#
# this code is a modification of
# https://pyimagesearch.com/2021/05/06/implementing-the-perceptron-neural-network-with-python/

import numpy as np

class Perceptron:
    
    # initialize Perceptron object
    # passed parameters:
    #   N     = number of inputs
    #   alpha = learning rate (0.1 = default)
    #
    def __init__(self, N, alpha=0.1):
		# initialize the weight matrix 
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        
        # store the learning rate
        self.alpha = alpha
        
    # activation function = step function
    # Returns 1 if x > 0, else returns 0
    def activation(self, x):
        return 1 if x > 0 else 0
    
    
    # train (learning) - adjust weights given training data
    # passed parameters:
    #   X - matrix containing inputs for each example in training data
    #   y - target value corresponding to each input X
    #   epochs - number of learning iterations 
    # result:  updates the weights self.W
    #
    def train(self, X, y, max_epochs=10):

		# loop over the desired number of epochs
        for n in np.arange(0, max_epochs):
            
            nmiss = 0
			# loop over each data vector
            for (x, target) in zip(X, y):

				# evaulate prediction of the neural net
                p = self.predict(x)
                error = p - target
                
                if error != 0:
                    nmiss = nmiss + 1
                
				# update the weight matrix
                self.W[1:] += -self.alpha * error * x    # input weights
                self.W[0]  += -self.alpha * error        # bias weight
                
            print(F"    training epoch {n:2}  # misses = {nmiss}")
            
            # if all training data is classified correctly, terminate training
            if nmiss == 0:
                break
                                
                    
    # predict - evaluate output of network given weights
    # passed parameters
    #   x - input values for one training set
    # take the dot product between the input features and the
	# weight matrix, add the bias and then pass the value through 
    # activation function
    #
    def predict(self, x):
        z = np.dot(x, self.W[1:]) + self.W[0]
        return self.activation(z)
    
    
    
# function that returns training data based on option
#
def train_data(opt):
    
    if opt == "OR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        return X,y
    
    elif opt == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return X,y
    
    elif opt == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        return X,y


#################   main program   ###################

# create training data
X,y = train_data("OR")

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], 0.1)
p.train(X, y, 20)

# now that our perceptron is trained we can evaluate it
# loop over the data points and compare network output to training data
print("[INFO] testing perceptron...")
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("    data={}, target={}, prediction={}".format(
		x, target[0], pred))
    
