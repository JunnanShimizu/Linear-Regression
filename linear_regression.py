'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Junnan Shimizu
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        ind = self.data.select_data(ind_vars)
        dep = self.data.select_data(dep_var)

        self.A = ind
        self.y = dep

        self.ind_vars = ind_vars
        self.dep_var = dep_var

        if method == 'scipy':
            self.linear_regression_scipy(ind, dep)
        elif method == 'normal':
            self.linear_regression_normal(ind, dep)
        elif method == 'qr':
            self.linear_regression_qr(ind, dep)

        y_pred = self.predict()

        self.residuals = self.compute_residuals(y_pred)
        self.r_squared(y_pred)

        pass

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''

        A = np.hstack([A, np.ones((np.shape(A)[0], 1))])

        x, residuals, rank, s = scipy.linalg.lstsq(A, y)

        num_ind_vars = np.shape(A)[1]

        self.slope = x[:num_ind_vars - 1, :1]
        self.intercept = x[num_ind_vars - 1, 0]
        
        return x
    
        pass

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A1 = np.hstack((A, np.ones((A.shape[0], 1))))
        At = np.transpose(A1)
        AtA = np.dot(At, A1)
        Aty = np.dot(At, y)

        AtA_inv = np.linalg.inv(AtA)

        c = np.dot(AtA_inv, Aty)

        num_samples = np.shape(c)[0]

        self.slope = c[:num_samples - 1, :1]
        self.intercept = c[1]

        return c

        pass

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''

        A = np.hstack([A, np.ones((np.shape(A)[0], 1))])
        Q, R = self.qr_decomposition(A)

        Qty = np.dot(np.transpose(Q), y)
        c = scipy.linalg.solve_triangular(R, Qty)

        num_samples = np.shape(c)[0]

        self.slope = c[:num_samples - 1, :1]
        self.intercept = c[1]

        return c

        pass

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''

        Q = np.zeros(shape=np.shape(A))

        for i in range(np.shape(A)[1]):
            j = A[:, i]
            for i2 in range(i):
                q = Q[:, i2]
                j = j - np.dot(q, j) * q
            Q[:, i] = j / np.linalg.norm(j)

        R = np.dot(np.transpose(Q), A)

        return Q, R

        pass

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''

        if np.shape(self.A)[1] == np.shape(self.slope)[0]:
            if X is None:
                result = np.dot(self.A, self.slope) + self.intercept
            else:
                result = np.dot(X, self.slope) + self.intercept
        else:
            if X is None:
                result = np.dot(self.A, self.slope.T) + self.intercept
            else:
                result = np.dot(X, self.slope) + self.intercept
            
        return result


        
        pass

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''

        y_mean = np.mean(self.y)

        sst = np.sum((self.y - y_mean) ** 2)
        sse = np.sum((self.y - y_pred) ** 2)

        ssr = sst - sse
        r2 = ssr / sst

        self.R2 = r2

        return r2

        pass

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''

        # print(self.y, '\n', y_pred)

        residuals = self.y - y_pred
        return residuals

        pass

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''

        return np.mean(np.square(self.compute_residuals(self.predict())))

        pass

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''

        line_x = np.linspace(self.A.min(), self.A.max(), 100)

        # line_y = ((self.slope * line_x) + self.intercept).T

        line_y = np.zeros(100)

        for i in range(self.p):
            line_y += self.slope[i] * line_x**(i + 1)

        line_y += self.intercept
        plot = plt.scatter(self.A, self.y, label='Data points')
        plt.plot(line_x, line_y, color='red', label='Linear regression')
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title + ' R^2: ' + str(self.R2))

        pass

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''

        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize=fig_sz, sharex='col', sharey='row')

        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                self.linear_regression(data_vars[i], data_vars[j])
                line_x = np.linspace(self.A.min(), self.A.max(), 100)
                line_y = ((self.slope * line_x) + self.intercept).reshape(100, 1)
                axes[i, j].set_title(round(self.R2, 5))
                if i == j and hists_on_diag == True:
                    axes[i, j].hist(self.data.data[:, i])
                else:
                    axes[i, j].scatter(self.data.data[:, j], self.data.data[:, i])
                    axes[i, j].plot(line_x, line_y, color='red', label='Linear regression')
                
        for i in range(len(data_vars)):
            axes[len(data_vars)-1, i].set_xlabel(data_vars[i])
            axes[i, 0].set_ylabel(data_vars[i])

        plt.show()

        pass

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''

        num_rows = np.shape(A)[0]

        poly = np.zeros((len(A), p + 1))

        for row in range(num_rows):
            for col in range(p + 1):
                poly[row, col] = A[row]**col
            
        poly = poly[:, 1:p+1]
        return poly

        pass

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''

        self.p = p
        ind = self.data.select_data(ind_var)
        dep = self.data.select_data(dep_var)

        self.A = ind
        self.y = dep

        poly = self.make_polynomial_matrix(ind, p)

        c = self.linear_regression_scipy(poly, dep)

        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.r_squared(y_pred)


        return c
    
        pass

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''

        return self.slope

        pass

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''

        return self.intercept

        pass

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''

        if ind_vars is not None:
            self.ind_vars = ind_vars
        if dep_var is not None:
            self.dep_var = dep_var
        if slope is not None:
            self.slope = slope
        if intercept is not None:
            self.intercept = intercept
        if p is not None:
            self.p = p

        pass
