import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 1
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

def one_hot (labels):
    X = np.zeros((len(labels), len(set(labels))))
    X[np.arange(len(labels)),labels - 1] = 1
    return X

def relu (x):
    return (x > 0) * x

def sigma (x):
    return 1. / (1. + np.exp(- x))

def softmax (z):
    znormed = z - np.max(z, axis=0, keepdims=True)
    denom = np.sum(np.exp(znormed), axis=0, keepdims=True)
    result = np.exp(znormed) / denom
    return result

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weights[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def forward_error (x, y, weights):
    Ws, bs = unpack(weights)

    zs = []
    hs = []
    h = x
    zs.append(x)  # Invalid, but doesn't matter because it won't be used
    hs.append(h)
    for i in range(NUM_HIDDEN_LAYERS):
        z = Ws[i].dot(h) + np.atleast_2d(bs[i]).T
        zs.append(z)
        h = relu(z)
        hs.append(h)

    yhat = softmax(Ws[-1].dot(h) + np.atleast_2d(bs[-1]).T)
    cost = - np.sum(y * np.log(yhat))
    if np.isnan(cost):
        1/0
    acc = np.mean(np.argmax(y, axis=0) == np.argmax(yhat, axis=0))
    
    # Return everything
    return cost, acc, zs, hs, Ws, yhat
   
def gradient (x, y, weights):
    _, _, zs, hs, Ws, yhat = forward_error(x, y, weights)  ### Correct -- capture W2

    dJdWs = []
    dJdbs = []
    g = yhat - y
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        dJdbs.append(np.sum(g, axis=1))
        if i < NUM_HIDDEN_LAYERS:
            dJdWs.append(g.dot(hs[i].T))
        else:
            dJdWs.append(g.dot(hs[i].T))
        g = Ws[i].T.dot(g)
        g = g * (zs[i] > 0)
    # Make sure to reverse the lists since we're doing *back*-propagation!
    dJdWs = dJdWs[::-1]
    dJdbs = dJdbs[::-1]

    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 

def show_filters (W):
    Ws,bs = unpack(W)
    plt.imshow(np.hstack([ np.pad(np.reshape(Ws[0][:,idx], [ 28, 28 ]), 2, mode='constant') for idx in range(NUM_HIDDEN) ]), cmap='gray'), plt.show()
 
def train (trainX, trainY, weights, testX, testY):
    NUM_EPOCHS = 1
    EPSILON = 1e-3
    BATCH_SIZE = 128
    for epoch in range(NUM_EPOCHS):
        print("epoch={}".format(epoch))
        for j in range(0, trainX.shape[1], BATCH_SIZE):
            idxs = range(j, min(j+BATCH_SIZE, trainX.shape[1]))
            minibatch = np.atleast_2d(trainX[:,idxs])
            minibatch += np.random.randn(minibatch.shape[0], minibatch.shape[1]) * 0.05
            grad = gradient(minibatch, np.atleast_2d(trainY[:,idxs]), weights)
            weights -= EPSILON * grad
        print("cost,acc={}".format(forward_error(testX, testY, weights)[0:2]))
    return weights

def initWeightsAndBiases ():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX = np.load("fashion_mnist_train_images.npy").T / 255.
        trainY = one_hot(np.load("fashion_mnist_train_labels.npy")).T
        testX = np.load("fashion_mnist_test_images.npy").T / 255.
        testY = one_hot(np.load("fashion_mnist_test_labels.npy")).T

    Ws, bs = initWeightsAndBiases()

    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    print(scipy.optimize.check_grad(lambda weights_: forward_error(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_)[0], \
                                    lambda weights_: gradient(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_), \
                                    weights))
    print(scipy.optimize.approx_fprime(weights, lambda weights_: forward_error(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), weights_)[0], 1e-6))

    weights = train(trainX, trainY, weights, testX, testY)
    #show_filters(weights)
