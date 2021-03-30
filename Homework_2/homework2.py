import os
import numpy as np
import math
import matplotlib.pyplot as plt
from pdb import set_trace
from torch.utils.tensorboard import SummaryWriter

def linear_regression(x, y):
	X_t = x.T
	X_y = X_t.dot(y) # x is a column vector so it needs to be transformed
	X_XT = X_t.dot(x)
	w = np.linalg.solve(X_XT, X_y) #(X_XT dot X_y)^-1
	return w


def model(x, w, b):
	y_hat = x.dot(w) + b
	return y_hat


def mean_square_error(y_hat, y):
	n = y.shape[0]
	mse = (1/(2 * n)) * np.sum(np.square(y_hat - y))
	return mse


def l2(w, y, alpha = 0.01):
	n = y.shape[0]
	wt_w = (w).dot(w.T) # (w^T)w
	reg = (1/(2 * n) * alpha) * wt_w # (alpha/2n) * ((w^T)w) 
	return reg


def regularized_mean_square_error(y, y_hat, reg):
	n = y.shape[0]
	mse = mean_square_error(y_hat, y)
	return mse + reg


#Train/Test Split performed by randomly taking inputs and their associated labels and assigning them to either a training group or a test group
#---------------------------------------------------#----------------#
#                                                   |                |
#                  Training set                     |   Testing Set  |
#                                                   |                |
#---------------------------------------------------#----------------#
#train_perc should be a value between 0 and 1, eg. train_perc 0.8
def random_train_test_split(dataset, train_perc = 0.8):
	perc_of_dataset = dataset.shape[1] * train_perc
	numpy.random.shuffle(dataset)

	train = dataset_arrx[:perc_of_dataset,:] 
	test = dataset_arr[perc_of_dataset:,:]

	return train, test


#Validation/Test Split performed by randomly taking inputs and their associated labels and assigning them to either a validation group or a test group
#---------------------------------------------------#----------------#
#                                                   |                |
#                  Training set                     |   Testing Set  |
#                                                   |                |
#---------------------------------------------------#----------------#
#                                                             V
#                                                |------------#---------#
#                                                |            |         |
#                                                | Validation | Testing |
#                                                |     Set    |   Set   |
#                                                |------------#---------#
#train_perc should be a value between 0 and 1, eg. train_perc 0.5 for a 50/50 split of the test set 
def random_test_validation_split(x, y, train_perc = 0.8):
	dataset_row_size, dataset_col_size = x.shape
	validation_set_size = int(dataset_row_size * train_perc)
	test_set_size = int(dataset_row_size - validation_set_size)

	#get random indices for both validation and test sets
	indices = np.random.permutation(x.shape[0])
	validation_p_index = indices[:validation_set_size]
	test_p_index = indices[test_set_size:]

	#validation and test labels
	validation_l = y[validation_p_index]
	test_l = y[test_p_index]

	#validation and test parameters 
	validation_p = x[validation_p_index]
	test_p = x[test_p_index]

	return validation_p, validation_l, test_p, test_l


#X is the input 
#y is the real result
#w is the weight 
#epsilon is the learning rate
#alpha is the regularized term
#reg_bool if you wish to use the regularized term
def gradient_descent(x, y, b, w, learning_rate, reg_bool, alpha = 0.1):
	# batch_size x 2304
	X_t = x.T # 2304 x batch_size
	n = y.shape[0]
	y_hat = model(x, w, b)

	#taking derivative 
	# 2304 x 1 
	derivative_of_w = (1/n) * X_t.dot(y_hat - y)
	derivative_of_b = (1/n) * np.sum(y_hat - y)

	# if regularized term is wanted then add that to the derivative with respect to w
	if reg_bool == True:
		derivative_of_w += ((alpha/n) * w)

	#this is gradient descent
	w = w - (learning_rate * derivative_of_w)
	b = b - (learning_rate * derivative_of_b)

	return w, b


#ripped from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(b))
	return a[p], b[p]


def visualize_that_beh(history: dict):
	# summarize history for accuracy
	plt.plot(history["epoch"], history["cost"], 'g', label='loss over epoch')
	plt.title('cost to epoch ')
	plt.ylabel('cost')
	plt.xlabel('epoch')
	plt.show()
	# summarize history for loss
	plt.plot(history["epoch"], history["learning_rate"], 'b', label='learning rate over epoch')
	plt.title('learning rate to epoch')
	plt.xlabel('Epochs')
	plt.ylabel('Learning Rate')
	plt.show()


def batch_iterator(x, y, size_of_batch):
	assert (len(x) == len(y))
	p = np.random.permutation(x.shape[0])
	x_rand = x[p]
	y_rand = y[p]
	for i in np.arange(0, x.shape[0], size_of_batch):
		yield x_rand[i:i + size_of_batch], y_rand[i:i + size_of_batch]


def stochastic_gradient_descent(x, y, b, w, learning_rate: float, num_of_epochs: int, size_of_batch: int, size_of_dataset: int, reg_bool: bool = True, alpha = 0.1):
	past_cost = 1e100
	current_cost = -1
	epochs_since_improved = 0
	writer = SummaryWriter()
	current_step = 0

	for e in range(num_of_epochs - 1):
		current_step += 1
		for mini_batch_x, mini_batch_y in batch_iterator(x, y, size_of_batch):

			y_hat = model(mini_batch_x, w, b)

			if reg_bool == True:
				reg = l2(w, mini_batch_y)
				current_cost = regularized_mean_square_error(mini_batch_y, y_hat, reg)
			else:
				current_cost = mean_square_error(y_hat, mini_batch_y)
				
			w, b = gradient_descent(mini_batch_x, mini_batch_y, b, w, learning_rate, reg_bool, alpha)

		entire_dataset_mse = mean_square_error(model(x, w, b), y)
		if math.isnan(entire_dataset_mse):
			break
		#print("MSE: ", entire_dataset)
		
	return w, b 

def find_lowest_loss(x, y, learning_rate, num_of_epochs, size_of_batch, reg_bool = True, alpha = 0.1):
	size_of_dataset = len(y)
	l = np.expand_dims(a=y, axis=-1)
	w_shape = (x.shape[1]) #2304 x 1
	w = np.random.rand(w_shape) * np.sqrt(1/(x.shape[1] + l.shape[1])) #set intial w value based off Xavier intilization
	b = np.random.rand(l.shape[1]) #set initial bias value
	reg = l2(w, y) # set regularized term
	w_new, b_new = stochastic_gradient_descent(x, y, b, w, learning_rate, num_of_epochs, size_of_batch, size_of_dataset, reg_bool, alpha)
	return w_new, b_new


def train_valid_test(x, w, b, y, reg: float, reg_bool: bool = False):
	y_hat = model(x, w, b)
	if reg_bool == True:
		validation_or_test = regularized_mean_square_error(y, y_hat, reg)
	else:
		validation_or_test = mean_square_error(y_hat, y)

	return validation_or_test


def load_data():
	#training set
	x_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48)) #2304 x 1
	y_tr = np.load("age_regression_ytr.npy")

	#testing set
	x_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
	y_te = np.load("age_regression_yte.npy")
	
	return x_tr, y_tr, x_te, y_te


def load_random_data():
	x = np.random.rand(500, 1)
	y = 4 * x + 3 + np.random.rand(500, 1)

	return x, y 

def grid_search():
	hyperparameters = {
		"learning_rate": [0.01, 0.001, 0.0001],
		"num_of_epochs": [100, 200, 300, 400, 500, 600],
		"size_of_batch": [1, 10, 50, 100, 200, 5000],
		"alpha": [0.9, 0.5, 0.1]	
	}
	for i in range(len(hyperparameters["num_of_epochs"])):
		for j in range(len(hyperparameters["size_of_batch"])):
			for k in range(len(hyperparameters["learning_rate"])):
				for l in range(len(hyperparameters["alpha"])):
						yield hyperparameters["learning_rate"][k], hyperparameters["num_of_epochs"][i], hyperparameters["size_of_batch"][j], hyperparameters["alpha"][l]

#Function Arguments:
#mse_bool is the boolean value that determines if mse will be performed without or with regularization, default is True
#alpha is the value for regularization, if used, default is 0.0
#ttv_val is the value associated with the type of split wanted. A Train/Test Split is 0, and a train/validation/split is 1, No Split Needed is 2. Default is 2
#learning rate is the hyperparameter associated with the gradient descent
def train_age_regressor(ttv_val: int = 2, the_set: int = 0, reg_bool = False):
	# Load data
	x_tr, y_tr, x_te, y_te = load_data()

	x_rand, y_rand = load_random_data()

	if ttv_val == 0:
		print("I made this before I realized train/test split is already given to you")

	elif ttv_val == 1:
		x_val, y_val, x_te, y_te = random_test_validation_split(x_te, y_te, train_perc = 0.8)

	
	best_loss = 1000000000
	best_learning_rate = -1
	best_num_of_epoch = -1
	best_size_of_batch = -1
	best_alpha = -1
	for learning_rate, num_of_epochs, size_of_batch, alpha in grid_search():

		w_trained, b_trained = find_lowest_loss(x_tr, y_tr, learning_rate, num_of_epochs, size_of_batch, reg_bool, alpha)
		reg = l2(w_trained, y_tr)
		check_loss = train_valid_test(x_te, w_trained, b_trained, y_te, reg, reg_bool)

		if math.isnan(check_loss):
			continue

		if the_set == 0: #perform test using weights and biases from training set on test set, thus getting loss on test set
			reg = l2(w_trained, y_te)
			loss = train_valid_test(x_te, w_trained, b_trained, y_te, reg, reg_bool)
			print("train/test loss: ", loss)

		elif the_set == 1: #perform test using weights and biases from training set on validation set, thus getting loss on validation set.
			reg = l2(w_trained, y_val)
			loss = train_valid_test(x_val, w_trained, b_trained, y_val, reg, reg_bool)
			print("train/validation loss: ", loss)

			if loss < best_loss:
				best_loss = loss
				best_learning_rate = learning_rate
				best_num_of_epoch = num_of_epochs
				best_size_of_batch = size_of_batch
				best_alpha = alpha

		else:
			print("you picked an incorrect the_set value")

	reg = l2(w_trained, y_te)
	loss = train_valid_test(x_te, w_trained, b_trained, y_te, reg, reg_bool)
	print("\n---------------------------------------------------------------------------------------------------------------------------------------------------------")
	print("train/validation/test loss: ", loss)
	print("reg_bool: ", reg_bool)
	print("best_loss: ", best_loss)
	print("best_learning_rate: ", best_learning_rate)
	print("best_num_of_epoch: ", best_num_of_epoch)
	print("best_size_of_batch: ", best_size_of_batch)
	print("best_alpha: ", best_alpha)
	print("--------------------------------------------------------------------------------------------------------------------------------------------------------- \n")

	return w_trained, b_trained


def main():
	os.chdir(os.path.dirname(os.path.abspath(__file__)))

	print("global based SGD using regularization")
	w_global, b_global = train_age_regressor(ttv_val = 1, the_set = 1, reg_bool = True)
	print("batch based SGD using regularization")
	w_batch, b_batch = train_age_regressor(ttv_val = 1, the_set = 1, reg_bool = True)

	print("global based SGD")
	w_global, b_global = train_age_regressor(ttv_val = 1, the_set = 1, reg_bool = False)
	print("batch based SGD")
	w_batch, b_batch = train_age_regressor(ttv_val = 1, the_set = 1, reg_bool = False)
	#print(np.sum(np.square(w_global - w_batch)))

if __name__ == '__main__':
	main()


