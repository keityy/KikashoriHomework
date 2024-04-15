import numpy as np
import matplotlib.pyplot as plt

point = np.loadtxt("original_point_set.txt")

mean_point = np.average(point,axis=1)

variance = np.cov(point)
#print(variance)

eigenvalues, eigenvectors = np.linalg.eig(variance)
print(f"eigenvalues: {eigenvalues} \n eigenvectors: {eigenvectors}")

smallest_eigenvalue = np.min(eigenvalues)
#print(smallest_eigenvalue)

index_of_smallest_eigenvalue = np.argmin(eigenvalues)
print(f"index of sallest eigenvalue{index_of_smallest_eigenvalue}")

smallest_eigenvector = eigenvectors[:, index_of_smallest_eigenvalue]
print(f"smallest eigenvector: {smallest_eigenvector}")

c = - np.dot(smallest_eigenvector,mean_point)
print(f"offset c:{c}")

x = point[0,:]
y = point[1,:]
# print(f"point:{point} \n x:{x} \n y:{y}")

X_MIN = 0
X_MAX = 10

def line_func(x,n,c):
    n1,n2=n[0],n[1]
    y = - (n1*x+c)/n2
    return y

line_point1=[X_MIN, ]

plt.scatter(x,y,c='k')
plt.plot([X_MIN, X_MAX], [line_func(X_MIN,smallest_eigenvector,c),line_func(X_MAX,smallest_eigenvector,c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
plt.show()