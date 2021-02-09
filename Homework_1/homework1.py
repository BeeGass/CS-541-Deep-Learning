import numpy as np
A_var = np.array([
    [5, 2, 1],
    [6, 8, 0],
    [2, 5, 8]
    ])
B_var = np.array([
    [5, 3, 2],
    [7, 9, 1],
    [3, 6, 9]
    ])
C_var = np.array([
    [6, 3, 2],
    [8, 0, 2],
    [4, 7, 0]
    ])


x = np.array([5, 2, 1, 6, 8])

x_1 = np.array([[5, 2, 1], 
                [6, 8, 8]])

x_2 = np.array([[5, 2, 1], 
                [6, 8, 8]])

x_3 = np.array([5, 2, 1])

y = np.array([5, 3, 2, 7, 9])

y_1 = np.array([[5, 3, 2], 
                [7, 9, 4]])

y_2 = np.array([[5, 3, 2], 
                [7, 9, 4]])

matrix = np.array([
    [1, 2, 0],
    [2, 2, 0],
    [3, 2, 0],
    [4, 2, 0]
    ])

matrix_two = np.array([
    [4, 3, 2, 1],
    [2, 2, 2, 2],
    [0, 0, 0, 0]
    ])

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return (np.dot(A, B) - C)

def problem_1c (A, B, C):
    return ((A * B) + C.T)

def problem_1d (x, y):
    return np.inner(x, y)

def problem_1e (A):
    return np.zeros(np.shape(A_var))

def problem_1f (A, x):
    return np.linalg.solve(A, x)

def problem_1g (A, x):
    return (np.linalg.solve(A.T, x.T)).T

def problem_1h (A, alpha):
    num_rows, num_cols = A.shape
    print("This is A \n", A)
    print("\nThis is I \n", np.eye(num_rows))
    return A + (np.eye(num_rows) * alpha)

def problem_1i (A, i):
    return np.sum(A[i, ::2])

def problem_1j (A, c, d):
    b = np.where(np.logical_and(c <= A, A <= d))
    return np.mean(A[b])

def problem_1k (A, k):
    eigen_value, eigen_vector = np.linalg.eig(A) #get eigenvalues and vectors
    descen_arr_indices = eigen_value.argsort() #sort indices in descending order
    matching_indices = descen_arr_indices[len(descen_arr_indices) - k:]
    return eigen_vector[:,matching_indices] #use indices to get respective eigen vectors 

def problem_1l (x, k, m, s):
    return np.random.multivariatenormal(x+m, np.eye(x.shape[0]) * s, k).T

def problem_1m (A):
    return np.random.permutation(A)

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
    m, n = np.shape(X)
    three_dim_mat = np.repeat(np.atleast_3d(X), n, axis=2)
    swap_mat = np.swapaxes(three_dim_mat, 1, 2)
    D = np.sqrt(np.sum(np.square(three_dim_mat-swap_mat), axis=0))
    return D

def linear_regression (X_tr, y_tr):
    X_t = X_tr.T
    X_y = np.matmul(X_tr.T, y_tr)
    X_XT = np.matmul(X_t, X_tr)
    return np.linalg.solve(X_XT, X_y)

def mean_square_error(y_hat, y):
    dist_diff = np.square(y_hat-y)
    err = np.mean(dist_diff)
    return err

def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)
    y_hat_tr = np.matmul(X_tr, w)
    y_hat_te = np.matmul(X_te, w)

    train = mean_square_error(y_hat_tr, ytr)
    test = mean_square_error(y_hat_te, yte)

    return train, test