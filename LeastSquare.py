import numpy as np
import matplotlib.pyplot as plt

# points = np.loadtxt("original_point_set.txt")
points = np.loadtxt("point_set_with_outliers.txt")
x = points[0,:]
y = points[1,:]
X_MIN = 0
X_MAX = 10

def least_square_estimation(points):
    mean_point = np.average(points,axis=1)
    variance = np.cov(points)
    eigenvalues, eigenvectors = np.linalg.eig(variance)
    smallest_eigenvalue = np.min(eigenvalues)
    index_of_smallest_eigenvalue = np.argmin(eigenvalues)
    smallest_eigenvector = eigenvectors[:, index_of_smallest_eigenvalue]
    c = - np.dot(smallest_eigenvector,mean_point)
    return smallest_eigenvector, c

def line_func(x,n,c):
    n1,n2=n[0],n[1]
    y = - (n1*x+c)/n2
    return y

smallest_eigenvector, c = least_square_estimation(points)

plt.scatter(x,y,c='k')
plt.plot([X_MIN, X_MAX], [line_func(X_MIN,smallest_eigenvector,c),line_func(X_MAX,smallest_eigenvector,c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
plt.show()