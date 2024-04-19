import numpy as np
import matplotlib.pyplot as plt

def line_func(x,n,c):
    n1,n2=n[0],n[1]
    y = - (n1*x+c)/n2
    return y

def gm_estimater(erros, e):
    sigma = np.std(erros)
    rho = e**2/(sigma**2+e**2)
    return rho

def weight_func_for_gm(errors, e):
    sigma = np.std(errors)
    weight = 2/(1+e**2/sigma**2)**2
    return weight

class RobustLineEstimator_GM:
    def __init__(self, points):
        self.transposed_points = np.transpose(points)
    
    def _error_func(self, n, c, x1, x2):
        n1, n2 = n[0], n[1]
        error = n1 * x1 + n2 * x2 + c
        return error
    
    def calc_error(self, n, c):
        errors = []
        x = self.transposed_points
        for x_k in x:
            x1, x2 = x_k[0], x_k[1]
            error = self._error_func(n, c, x1, x2)
            errors.append(error)
        return errors

    def weighted_points(self, errors):
        weighted_points = np.array([weight_func_for_gm(errors, errors[k]) * self.transposed_points[k] for k in range(len(self.transposed_points))])
        return weighted_points
    
    def weighted_errors(self, errors):
        weighted_errors = [weight_func_for_gm(errors, errors[k]) for k in range(len(errors))]
        return weighted_errors
    
    def calc_weighted_covariance_matrix(self,errors):
        weighted_points = self.weighted_points(errors)
        sum_of_weightedpoints = np.sum(weighted_points, axis=0)

        weighted_errors = self.weighted_errors(errors)
        # print(f"weighted_errors:{weighted_errors}")

        sum_of_weighted_errors = np.sum(weighted_errors)

        weighted_mean = sum_of_weightedpoints / sum_of_weighted_errors

        deviation_from_mean = weighted_points - weighted_mean

        for k in range(len(errors)):
            sumof_weighted_deviation_matrix = weighted_errors[k] * (deviation_from_mean[k].reshape(2,1) @ deviation_from_mean[k].reshape(1,2))

        weighted_covariance_matrix = sumof_weighted_deviation_matrix / sum_of_weighted_errors

        return weighted_mean, weighted_covariance_matrix

    def calc_smallest_eigenvector(self, weighted_covariance_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(weighted_covariance_matrix)

        smallest_eigenvalue = np.min(eigenvalues)

        index_of_smallest_eigenvalue = np.argmin(eigenvalues)

        smallest_eigenvector = eigenvectors[:, index_of_smallest_eigenvalue]

        return smallest_eigenvector
    
    def calc_offset(self, smallest_eigenvector, weighted_mean):
        self.c = - np.dot(smallest_eigenvector,weighted_mean)
        return - np.dot(smallest_eigenvector,weighted_mean)
    
    def calc_estimation_line(self, smallest_eigenvector, c):
        errors = self.calc_error(smallest_eigenvector, c)
        weighted_mean ,weighted_covariance_matrix = self.calc_weighted_covariance_matrix(errors)
        smallest_eigenvector = self.calc_smallest_eigenvector(weighted_covariance_matrix)
        c = self.calc_offset(smallest_eigenvector, weighted_mean)
        return smallest_eigenvector, c

if __name__ == "__main__":
    points = np.loadtxt("point_set_with_outliers.txt")
    
    #used for plotting
    x = points[0,:]
    y = points[1,:]
    X_MIN = 0
    X_MAX = 10

    #Least square method
    mean_point = np.average(points,axis=1)

    variance = np.cov(points)

    eigenvalues, eigenvectors = np.linalg.eig(variance)
    smallest_eigenvalue = np.min(eigenvalues)
    index_of_smallest_eigenvalue = np.argmin(eigenvalues)

    ls_smallest_eigenvector = eigenvectors[:, index_of_smallest_eigenvalue]
    ls_c = - np.dot(ls_smallest_eigenvector,mean_point)
    
    #GM estimation
    estimater = RobustLineEstimator_GM(points)
    for i in range(10):
        if i == 0:
            smallest_eigenvector = ls_smallest_eigenvector
            c = ls_c
            lines = []
        else:
            smallest_eigenvector, c = estimater.calc_estimation_line(smallest_eigenvector, c)
            print(f"i:{i} \n smallest_eigenvector:{smallest_eigenvector} \n c:{c}")
    plt.scatter(x,y,c='k')
    plt.plot([X_MIN, X_MAX], [line_func(X_MIN, smallest_eigenvector, c),line_func(X_MAX, smallest_eigenvector, c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
    plt.show()
    plt.close()
