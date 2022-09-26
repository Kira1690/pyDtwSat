from inspect import stack
import math
from re import M
from this import s
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import path

from timeseries_patterns import *

def mlwf(alpha,beta, t_i,t_j):
    g = abs(int(t_i) - int(t_j))
    a = -alpha*(g - beta)
    exp = math.exp(a)
    
    omaga = 1/ (1 + exp)

    return omaga

def dist_matrix_dtw(x,y):
    N = x.shape[0]
    M = y.shape[0]
    #print(x.shape[0], y.shape[0])
    dist_mat_dtw = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            dist_mat_dtw[i,j] = abs(x[i] - y[j])
    
    return dist_mat_dtw

def dist_matrix_dtw_window(x,y,window):
    N = x.shape[0]
    M = y.shape[0]
    W = np.max([window, abs(N-M)])

    dist_mat_dtw_window = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            dist_mat_dtw_window[i,j] = np.inf
    dist_mat_dtw_window[0,0] = 0

    for i in range(N):
        for j in range(np.max([1,i-W]), np.min([M, i+W])):
            dist_mat_dtw_window[i,j] = abs(x[i] - y[j])

    return dist_mat_dtw_window


def dist_matrix_twdtw(x,y,t_x,t_y,alpha,beta):
    N = x.shape[0]
    M = y.shape[0]
    #print(N,M)
    dist_mat_twdtw = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            dist_mat_twdtw[i,j] = mlwf(alpha,beta,t_x[i],t_y[j]) + abs(x[i] - y[j])

    return dist_mat_twdtw

def dist_matrix_twdtw_window(x,y,window,t_x,t_y,alpha,beta):
    N = x.shape[0]
    M = y.shape[0]
    W = np.max([window, abs(N-M)])
    dist_mat_twdtw_window = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            dist_mat_twdtw_window[i,j] = np.inf
    dist_mat_twdtw_window[0,0] = 0

    for i in range(N):
        for j in range(np.max([1,i-W]), np.min([M, i+W])):
            dist_mat_twdtw_window[i,j] = mlwf(alpha,beta,t_x[i],t_y[j]) + abs(x[i] - y[j])

    return dist_mat_twdtw_window

def dp(dist_mat):

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

def get_stack_distmat_dtw(y,stack_array ):
    S,N,M = stack_array.shape
    stack_dist_mat_dtw = np.zeros((N,M), dtype = object)

    for i in range(N):
        for j in range(M):
            ts_pixel = np.zeros((S))
            for s in range(S):
                ts_pixel[s] = stack_array[s,i,j]
            dist_mat_dtw =  dist_matrix_dtw(ts_pixel, y)
            stack_dist_mat_dtw[i,j] = dist_mat_dtw
    
    return stack_dist_mat_dtw

def get_stack_distmay_dtw_window(y, stack_array, window):
    S,N,M = stack_array.shape
    stack_dist_mat_dtw = np.zeros((N,M), dtype = object)

    for i in range(N):
        for j in range(M):
            ts_pixel = np.zeros((S))
            for s in range(S):
                ts_pixel[s] = stack_array[s,i,j]
            dist_mat_dtw =  dist_matrix_dtw_window(ts_pixel, y, window)
            stack_dist_mat_dtw[i,j] = dist_mat_dtw
    
    return stack_dist_mat_dtw

def get_stack_distmat_twdtw(y, stack_array,t_stack,t_y,alpha,beta):
    S,N,M = stack_array.shape
    stack_dist_mat_twdtw = np.zeros((N,M), dtype=object)
 
    for i in range(N):
        for j in range(M):
            ts_pixel = np.zeros((S))
            for s in range(S):
                ts_pixel[s] = stack_array[s,i,j]
            dist_mat_twdtw = dist_matrix_twdtw(ts_pixel,y,t_stack,t_y,alpha,beta)
            stack_dist_mat_twdtw[i,j] = dist_mat_twdtw

    return stack_dist_mat_twdtw

def get_stack_distmat_twdtw_window(y, stack_array,window,t_stack,t_y,alpha,beta):
    S,N,M = stack_array.shape
    stack_dist_mat_twdtw = np.zeros((N,M), dtype=object)
 
    for i in range(N):
        for j in range(M):
            ts_pixel = np.zeros((S))
            for s in range(S):
                ts_pixel[s] = stack_array[s,i,j]
            dist_mat_twdtw = dist_matrix_twdtw_window(ts_pixel,y,window,t_stack,t_y,alpha,beta)
            stack_dist_mat_twdtw[i,j] = dist_mat_twdtw

    return stack_dist_mat_twdtw    


def get_costmat(dist_mat):
    N,M = dist_mat.shape
    cost_mat = np.zeros((N,M), dtype= object)
    path_mat = np.zeros((N,M), dtype=object)
    for i in range(N):
        for j in range(M):
            path_mat[i,j], cost_mat[i,j] = dp(dist_mat[i,j])


    return path_mat, cost_mat



def alignment_cost_matrix(cost_matrix, path_mat):
    N,M = cost_matrix.shape
    alignment_mat = np.zeros((N,M), dtype = object)

    for i in range(N):
        for j in range(M):
            cost_mat = cost_matrix[i,j]
            path = path_mat[i,j]
            alignment = []
            for p in path:
                path_val = cost_mat[p[0],p[1]]
                alignment.append(path_val)
            alignment_val = np.nanmean(np.array(alignment))   
            alignment_mat[i,j] = alignment_val
    return alignment_mat


