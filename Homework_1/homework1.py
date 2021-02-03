import numpy as np

var_a = np.array([5, 2, 1, 6, 8, 0, 2, 5, 8])
A_var = var_a.reshape((3, 3))

var_b = np.array([5, 3, 2, 7, 9, 1, 3, 6, 9])
B_var = var_b.reshape((3, 3))

var_c = np.array([6, 3, 2, 8, 0, 2, 4, 7, 0])
C_var = var_c.reshape((3, 3))

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return (np.dot(A, B)) - C

def problem_1c (A, B, C):
    return (A * B) + (np.transpose(C))

def problem_1d (x, y):
    return np.dot(np.transpose(x), y)

def problem_1e (A):
    return np.zeros(A)

def problem_1f (A, x):
    return np.linalg.solve(A, x)

def problem_1g (A, x):
    return np.linalg.solve(x, A)

def problem_1h (A, alpha):
    num_rows, num_cols = A.shape
    return A + (np.eye(num_rows) * alpha)

def problem_1i (A, i):
    return np.sum(A[i, 1::2])

def problem_1j (A, c, d):
    b = np.where((c <= A or A <= d))
    return np.mean(A[b])

def problem_1k (A, k):
    eigen_value, eigen_vector = np.linalg.eig(A)
    descen_arr_indices = eigen_value[::-1].argsort()
    return eigen_vector[:,descen_arr_indices[:,:k]]

def problem_1l (x, k, m, s):
    x_row, x_col  = x.shape()
    m_z =  m * np.ones(x.shape)
    s_i = s * np.eye(x_row, k)
    n_k = np.random.multivariatenormal(x + m_z, s_i)
    return n_k

def problem_1m (A):
    return np.random.shuffle(A)

def problem_1n (x):
    x_bar = np.mean(x)
    sigma = np.std(x)
    y_vec = []
    for i in range(len(x)):
        x_i = x[i]
        y_i =  (x_i - x_bar) / sigma 
        y_vec.append(y_i)

    return y_vec

def problem_1o (x, k):
    return np.repeat(np.atleast_2d(x),k)

def problem_1p (X):
    return ...

def linear_regression (X_tr, y_tr):
    ...

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
