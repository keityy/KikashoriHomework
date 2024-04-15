import numpy as np
import matplotlib.pyplot as plt

points = np.loadtxt("point_set_with_outliers.txt")

mean_point = np.average(points,axis=1)

variance = np.cov(points)
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

x = points[0,:]
y = points[1,:]
# print(f"point:{point} \n x:{x} \n y:{y}")

X_MIN = 0
X_MAX = 10

def line_func(x,n,c):
    n1,n2=n[0],n[1]
    y = - (n1*x+c)/n2
    return y

class CalcError:
    def __init__(self, points):
        #transpose points
        self.points = np.transpose(points)

    def _error_func(self, n, c, x1, x2):
        n1, n2 = n[0], n[1]
        error = n1 * x1 + n2 * x2 + c
        return error
    
    def calc(self, n, c,):
        errors = []
        x = self.points
        for x_k in x:
            x1, x2 = x_k[0], x_k[1]
            error = self._error_func(n, c, x1, x2)
            errors.append(error)
        return errors

def gm_estimater(erros, e):
    sigma = np.std(erros)
    rho = e**2/(sigma**2+e**2)
    return rho

def weight_func_for_gm(errors, e):
    sigma = np.std(errors)
    weight = 2/(1+e**2/sigma**2)
    return weight


transposed_points = np.transpose(points)
print(f"len(points):{len(points)} \n len(transposed_points):{len(transposed_points)}")

errors = CalcError(points).calc(smallest_eigenvector, c)
print(f"errors:{errors} \n len(erros):{len(errors)}")

print(f"total:{np.sum(transposed_points, axis=0)}")
weight_point1= weight_func_for_gm(errors, errors[0]) * transposed_points[0]
print(f"weight_point1:{weight_point1}")

weighted_points = [weight_func_for_gm(errors, errors[k]) * transposed_points[k] for k in range(len(points))]
print(f"weighted_points:{weighted_points}")


# weighted_mean = [weight_func_for_gm(errors, errors[k]) * transposed_points[k] for k in range(len(points))]
# print(weighted_mean)

# plt.scatter(x,y,c='k')
# plt.plot([X_MIN, X_MAX], [line_func(X_MIN,smallest_eigenvector,c),line_func(X_MAX,smallest_eigenvector,c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
# plt.show()