"""
Fit circle to set of points. 
Find radius of curvature. 
Find planitude and alatude with respect to another point.
"""
import numpy as np
from scipy.optimize import minimize

def read_points_from_ds9_file():
    return None

def mean_radius(x, y, xc, yc):
    return np.mean(np.hypot(x - xc, y - yc))

def square_deviation(x, y, xc, yc):
    rm = mean_radius(x, y, xc, yc)
    return np.sum((np.hypot(x - xc, y - yc) - rm)**2)

def objective_f(center, xdata, ydata):
    """Function for minimize"""
    return square_deviation(xdata, ydata, center[0], center[1])
    
def fit_circle_to_xy(x, y):
    # guess the starting values
    soln0 = np.array((np.mean(x), np.mean(y)))
    # Do the fitting
    soln = minimize(objective_f, soln0, args=(x, y))
    return soln
    

TESTDATA = np.array([1, 2, 3, 4]), np.array([1, 2, 2, 1])
TESTCENTER = np.array([2.5, 0.5])

if __name__ == "__main__":
    results = fit_circle_to_xy(*TESTDATA)
    assert np.allclose(results.x, TESTCENTER)
    print(results)
