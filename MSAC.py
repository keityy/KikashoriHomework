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

def ls_estimator(points):
    #Least square method
    mean_point = np.average(points,axis=1)
    variance = np.cov(points)
    eigenvalues, eigenvectors = np.linalg.eig(variance)
    smallest_eigenvalue = np.min(eigenvalues)
    index_of_smallest_eigenvalue = np.argmin(eigenvalues)
    smallest_eigenvector = eigenvectors[:, index_of_smallest_eigenvalue]
    c = - np.dot(smallest_eigenvector,mean_point)
    return smallest_eigenvector, c

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

class CalcRANSAC:
    def __init__(self, points, delta, max_iterations):
        self.transposed_points = np.transpose(points)
        self.delta = delta # threshold for inliers
        self.max_iterations = max_iterations
    
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
    
    def process_calculation(self):
        # RANSAC
        # 1. Select random pare of points
        # 2. Fit a line to the selected points
        # 3. Calculate the error for all points
        # 4. Select and count inliers
        x=self.transposed_points
        for i in range(self.max_iterations):
            if i == 0:
                biggest_inliers_num = 0
                biggest_inliers = []
            random_points_index = np.random.choice(range(len(self.transposed_points)), 2)
            points = [self.transposed_points[random_points_index[0]], self.transposed_points[random_points_index[1]]]
            points_transposed = np.transpose(points)
            n,c = ls_estimator(points_transposed)
            errors = self.calc_error(n, c)
            inliers_index = []
            for i in range(len(errors)):
                if abs(errors[i]) < self.delta:
                    inliers_index.append(i)
            inliers = np.array(inliers_index)
            if len(inliers) > biggest_inliers_num:
                biggest_inliers_num = len(inliers)
                biggest_inliers = inliers
                best_n = n
                best_c = c
        biggest_inlier_points = np.array([self.transposed_points[biggest_inliers[i]] for i in range(len(biggest_inliers))])
        return best_n, best_c, biggest_inlier_points

class CalcMSAC:
    def __init__(self, points, delta, max_iterations):
        self.transposed_points = np.transpose(points)
        self.delta = delta # threshold for inliers
        self.max_iterations = max_iterations
    
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
    
    def _loss_func(self, error):
        if abs(error) > self.delta:
            loss = self.delta**2
        else:
            loss = error**2
        return loss

    def calc_total_loss(self, errors):
        loss = 0
        for error in errors:
            loss += self._loss_func(error)
        return loss
    
    def calculation(self):
        # MSAC
        # 1. Select random pare of points
        # 2. Fit a line to the selected points
        # 3. Calculate the error and loss for all points
        # 4. Update the best model
        x=self.transposed_points
        for i in range(self.max_iterations):
            if i == 0:
                biggest_total_loss = 100000
            random_points_index = np.random.choice(range(len(self.transposed_points)), 2)
            points = [self.transposed_points[random_points_index[0]], self.transposed_points[random_points_index[1]]]
            points_transposed = np.transpose(points)
            n,c = ls_estimator(points_transposed)
            errors = self.calc_error(n, c)
            total_loss = self.calc_total_loss(errors)
            if total_loss < biggest_total_loss:
                biggest_total_loss = total_loss
                best_n = n
                best_c = c
        return best_n, best_c, biggest_total_loss

if __name__ == "__main__":
    points = np.loadtxt("point_set_with_outliers.txt")
    estimation_type="GM" # chose from "LS", "GM", "RANSAC", "MSAC"
    
    #used for plotting
    x = points[0,:]
    y = points[1,:]
    X_MIN = 0
    X_MAX = 10

    #Least square method
    if estimation_type == "LS":
        ls_smallest_eigenvector, ls_c = ls_estimator(points)
    
    #GM estimation
    if estimation_type == "GM":
        ls_smallest_eigenvector, ls_c = ls_estimator(points)
        estimator = RobustLineEstimator_GM(points)
        for i in range(10):
            if i == 0:
                smallest_eigenvector = ls_smallest_eigenvector
                c = ls_c
                lines = []
            else:
                smallest_eigenvector, c = estimator.calc_estimation_line(smallest_eigenvector, c)
                print(f"i:{i} \n smallest_eigenvector:{smallest_eigenvector} \n c:{c}")
    
    #RANSAC
    if estimation_type == "RANSAC":
        delta = 1
        # number of iterations
        # inlier_rate>0.904 sample_size=2 なので，3回以上で最適な結果が得られる確率が99%以上  
        max_iterations = 3
        np.random.seed(33)#64 33
        ransac = CalcRANSAC(points, delta, max_iterations)
        ransac_n, ransac_c, biggest_inlier_points = ransac.process_calculation()
        biggest_inlier_points = np.transpose(biggest_inlier_points)

    #MSAC
    if estimation_type == "MSAC":
        delta = 1
        max_iterations = 100
        np.random.seed(0)
        msac = CalcMSAC(points, delta, max_iterations)
        msac_n, msac_c, biggest_total_loss = msac.calculation()
    
    plt.scatter(x,y,c='k')

    if estimation_type == "LS":
        plt.plot([X_MIN, X_MAX], [line_func(X_MIN,ls_smallest_eigenvector,ls_c),line_func(X_MAX,ls_smallest_eigenvector,ls_c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
        #set graph title
        plt.title(f"Estimated line by LS")

    if estimation_type == "GM":
    #plot the estimated line by GM (IRLS)
        plt.plot([X_MIN, X_MAX], [line_func(X_MIN, smallest_eigenvector, c),line_func(X_MAX, smallest_eigenvector, c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
        #set graph title
        plt.title(f"Estimated line by GM (IRLS)")

    if estimation_type == "RANSAC":
        #plot the estimated line by RANSAC
        plt.plot([X_MIN, X_MAX], [line_func(X_MIN, ransac_n, ransac_c),line_func(X_MAX, ransac_n, ransac_c)]) #(0, b)地点から(xの最大値,ax + b)地点までの線
        #plot the area of inliers as dotted line
        plt.plot([X_MIN, X_MAX], [line_func(X_MIN, ransac_n, ransac_c)+delta,line_func(X_MAX, ransac_n, ransac_c)+delta], linestyle='dotted')
        plt.plot([X_MIN, X_MAX], [line_func(X_MIN, ransac_n, ransac_c)-delta,line_func(X_MAX, ransac_n, ransac_c)-delta], linestyle='dotted')
        #plot inliers as red dots
        plt.scatter(biggest_inlier_points[0,:],biggest_inlier_points[1,:],c='r')
        #set graph title
        plt.title(f"Estimated line by RANSAC")


    if estimation_type == "MSAC":
        #plot the estimated line by MSAC
        plt.plot([X_MIN, X_MAX], [line_func(X_MIN, msac_n, msac_c),line_func(X_MAX, msac_n, msac_c)])
        #set graph title
        plt.title(f"Estimated line by MSAC")

    plt.show()
    plt.close()
