# +++ imports +++

from typing import Union, Tuple
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# +++ basic functions for fitting +++

def line(x:np.ndarray, m:float, b:float) -> np.ndarray:
    return m*x + b

def quadratic(x:np.ndarray, x_ext:float, a:float, b:float) -> np.ndarray:
    return a*(x-x_ext)**2 + b

def sin(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' A general sine funcition for fitting. Note that both parameters x and b are in degrees.

    Returns
    -------
    float
        a * sin(x + b) + c
    '''
    return a * np.sin(np.deg2rad(x + b)) + c

def sin_sq(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' A general sine squared funcition for fitting. Note that both parameters x and b are in degrees.

    Returns
    -------
    float
        a * sin^2(x + b) + c
    '''
    return a * np.sin(np.deg2rad(x + b))**2 + c

def sin2(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' A general sine(2x) funcition for fitting. Note that both parameters x and b are in degrees.

    Returns
    -------
    float
        a * sin(2*x + b) + c
    '''
    return a * np.sin(np.deg2rad(2*x + b)) + c

def cos(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' A general cosine funcition for fitting. Note that both parameters x and b are in degrees.

    Returns
    -------
    float
        a * cos(x + b) + c
    '''
    return a * np.cos(np.deg2rad(x + b)) + c

def cos_sq(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' A general cosine squared funcition for fitting. Note that both parameters x and b are in degrees.

    Returns
    -------
    float
        a * cos^2(x + b) + c
    '''
    return a * np.cos(np.deg2rad(x + b))**2 + c

def cos2(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' A general cosine(2x) funcition for fitting. Note that both parameters x and b are in degrees.

    Returns
    -------
    float
        a * cos(2*x + b) + c
    '''
    return a * np.cos(np.deg2rad(2*x + b)) + c

def sec(x:np.ndarray, a:float, b:float) -> np.ndarray:
    ''' Secant fit function.

    Returns
    -------
    float
        a / cos(x) + b
    '''
    return a / np.cos(np.deg2rad(x)) + b

# dictionary

FIT_FUNCS = {
    'linear': line,
    'line': line,
    'quad': quadratic,
    'quadratic': quadratic,
    'sin': sin,
    'sin_sq': sin_sq,
    'sin2': sin2,
    'cos': cos,
    'cos_sq': cos_sq,
    'cos2': cos2,
    'sec': sec
}

# +++ functions for fitting +++

def eval(func:Union[str, 'function'], **kwargs):
    ''' Evaluate a fit function with given arguments. '''
    if isinstance(func, str):
        func = FIT_FUNCS[func]
    return func(**kwargs)

def fit(func:Union[str,'function'], x:np.ndarray, y:np.ndarray,  y_err:np.ndarray, full_covm:bool=False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    ''' Fit data to a sine function

    Parameters
    ----------
    x : np.ndarray
        X values
    y : np.ndarray
        Y values
    y_err : np.ndarray
        Y errors
    full_covm : bool, optional (default=False)
        If true, a full covariance matrix will be returned rather than individual SEMs.
    **kwargs
        Get passed to scipy.optimize.curve_fit
    
    Returns
    -------
    np.ndarray
        Fit parameters.
    np.ndarray
        SEM uncertainties, or if full_covm=True, the full covariance matrix.
    '''
    # get the function to fit
    if isinstance(func, str):
        func = FIT_FUNCS[func]
    # fit the function
    popt, pcov = opt.curve_fit(func, x, y, sigma=y_err, absolute_sigma=True, **kwargs)
    if full_covm:
        return popt, pcov
    else:
        return popt, np.sqrt(np.diag(pcov))

def plot_func(func:Union[str,'function'], args:np.ndarray, x:Union[tuple,np.ndarray], ax:plt.Axes=None, num_points:int=300, **kwargs) -> None:
    ''' Plot a function
    
    Parameters
    ----------
    func : Union[str,function]
        Function to plot, or name of function to use for plotting.
    args : tuple
        Additional arguments for the function.
    x : np.ndarray
        X values. Function will be plotted from the minimum to the maximum of this range.
    ax : plt.Axes, optional
        Axes to plot on, by default None.
    num_points : int, optional
        Number of points to plot, by default 300.
    **kwargs : dict
        Additional arguments for plt/ax.plot().
    '''
    # get the function to plot
    if isinstance(func, str):
        func = FIT_FUNCS[func]
    # get axes
    if ax is None:
        ax = plt.gca()
    # plot the function
    x = np.linspace(np.min(x), np.max(x), num_points)
    ax.plot(x, func(x, *args), **kwargs)

def find_ratio(func1:Union[str,'function'], args1:np.ndarray, func2:Union[str,'function'], args2:np.ndarray, pct_1:float, x:Union[tuple,np.ndarray], guess:float=None):
    ''' Find where the two functions satisfy func1(x) / (func1(x) + func2(x)) = pct_1

    Parameters
    ----------
    func1 : Union[str,function]
        First function.
    args1 : np.ndarray
        Additional arguments for the first function.
    func2 : Union[str,function]
        Second function.
    args2 : np.ndarray
        Additional arguments for the second function.
    pct_1 : float
        Percentage of the first function.
    x : Union[tuple,np.ndarray]
        X values, the search will be limited within the minimum and maximum of this range.
    guess : float, optional
        Initial guess for the x value, by default average of the minimum and maximum of x.
    '''
    # get the functions
    if isinstance(func1, str):
        func1 = FIT_FUNCS[func1]
    if isinstance(func2, str):
        func2 = FIT_FUNCS[func2]
    # get the range
    x_min, x_max = np.min(x), np.max(x)
    # get the initial guess
    if guess is None:
        guess = (x_min + x_max) / 2
    # get the function to minimize
    def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple) -> float:
        return np.abs((1-pct_1)*func1(x_, *args1_) - pct_1*func2(x_, *args2_))
    # minimize
    res = opt.minimize(min_me, guess, args=(args1, args2), bounds=((x_min, x_max),))
    # obtain the result
    return res.x[0]

def find_value(func:Union[str,'function'], kwargs:dict, target:float):

    # define a function to minimize
    def min_me(x_:np.ndarray) -> float:
        return np.abs(eval(func, x=x_, **kwargs) - target)
    # minimize it
    res = opt.minimize(min_me, 10)
    # return the x value that minimizes it
    return res.x[0]
