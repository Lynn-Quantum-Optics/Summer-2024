# +++ imports +++

from typing import Union, Tuple
import numpy as np
import uncertainties.unumpy as unp
import uncertainties.core as ucore
from uncertainties import ufloat
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib

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

def sin2_sq(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' Sin(2x)^2 fit function

    Returns
    -------
    float
        a * sin^2(2*x + b) + c
    '''

    return a * (np.sin(np.deg2rad(2*x + b)))**2 + c

def cos2_sq(x:np.ndarray, a:float, b:float, c:float) -> np.ndarray:
    ''' Cos(2x)^2 fit function

    Returns
    -------
    float
        a * cos^2(2*x + b) + c
    '''

    return a * (np.cos(np.deg2rad(2*x + b)))**2 + c

def quartic(x:np.ndarray, a:float, b:float, c:float, d:float, e:float) -> np.ndarray:
    ''' A quartic polynomial fit function

    Returns
    -------
    float
        a * x^4 + b * x^3 + c * x^2 + d * x + e
    '''
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

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
    'sec': sec,
    'sin2_sq': sin2_sq,
    'cos2_sq': cos2_sq,
    'quartic': quartic
}

# +++ functions for fitting +++

def evalkw(func:Union[str, 'function'], **kwargs):
    ''' Evaluate a fit function with given arguments.
    
    Parameters
    ----------
    func : Union[str, 'function']
        The function to evaluate. If a string, it will be looked up in FIT_FUNCS.
    **kwargs : dict[str, Union[np.ndarray, float, ufloat]]
        The arguments to pass to the function. All ufloats will be converted to floats.
    '''
    # get the function
    if isinstance(func, str):
        func = FIT_FUNCS[func]
    # convert kwargs to floats
    for k, v in kwargs.items():
        # convert arrays to float arrays
        if isinstance(v, np.ndarray) and v.dtype == np.dtype('O'):
            kwargs[k] = unp.nominal_values(v)
        # convert ufloats to floats
        if isinstance(v, ucore.Variable):
            kwargs[k] = v.nominal_value
    # evaluate the function
    return func(**kwargs)

def eval(func:Union[str, 'function'], *args):
    ''' Evaluate a fit function with given arguments.
    
    Parameters
    ----------
    func : Union[str, 'function']
        The function to evaluate. If a string, it will be looked up in FIT_FUNCS.
    *args : Tuple[Union[np.ndarray, float, ufloat]]
        The arguments to pass to the function. All ufloats will be converted to floats.
    '''
    # get the function
    if isinstance(func, str):
        func = FIT_FUNCS[func]
    # convert kwargs to floats
    new_args = []
    for v in args:
        # convert arrays to float arrays
        if isinstance(v, np.ndarray) and v.dtype == np.dtype('O'):
            new_args.append(unp.nominal_values(v))
        elif isinstance(v, ucore.Variable):
            # convert ufloats to floats
            new_args.append(v.nominal_value)
        else:
            # no modification needed
            new_args.append(v)
    # evaluate the function
    return func(*new_args)

def fit(func:Union[str,'function'], x:np.ndarray, y:np.ndarray, full_covm:bool=False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    ''' Fit data to a sine function

    Parameters
    ----------
    x : np.ndarray
        X values
    y : np.ndarray
        Y values and uncertainties in the form of a uarray.
    full_covm : bool, optional (default=False)
        If true, a full covariance matrix will be returned rather than individual SEMs.
    **kwargs
        Get passed to scipy.optimize.curve_fit
    
    Returns
    -------
    np.ndarray
        Fit parameters with SEM uncertainties in the form of a uarray.
    or if full_covm=True
    np.ndarray, np.ndarray
        Fit parameters and the full covariance matrix.
    '''
    # get the function to fit
    if isinstance(func, str):
        func = FIT_FUNCS[func]
    # fit the function
    popt, pcov = opt.curve_fit(
        f=func,
        xdata=x,
        ydata=unp.nominal_values(y), sigma=unp.std_devs(y),absolute_sigma=True,
        **kwargs)
    # return in the correct format
    if full_covm:
        return popt, pcov
    else:
        return unp.uarray(popt, np.sqrt(np.diag(pcov)))

def plot_errorbar(x:np.ndarray, y:np.ndarray, ax:plt.Axes=None, **kwargs) -> matplotlib.container.ErrorbarContainer:
    ''' Plot errorbar data using ufloats.
    
    Parameters
    ----------
    x : np.ndarray
        X values.
    y : np.ndarray
        Y values and uncertainties in the form of a uarray.
    ax : plt.Axes, optional
        Axes to plot on, by default the current axes will be used.
    **kwargs : dict
        Additional arguments for plt/ax.errorbar().
    
    Returns
    -------
    matplotlib.container.ErrorbarContainer
        The errorbar container.
    '''
    # get axes
    if ax is None:
        ax = plt.gca()
    # plot the data
    return ax.errorbar(x=x, y=unp.nominal_values(y), yerr=unp.std_devs(y), **kwargs)

def plot_func(func:Union[str,'function'], args:np.ndarray, x:Union[tuple,np.ndarray], ax:plt.Axes=None, num_points:int=300, **kwargs) -> matplotlib.lines.Line2D:
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
    
    Returns
    -------
    matplotlib.lines.Line2D
        The line object.
    '''
    # get axes
    if ax is None:
        ax = plt.gca()
    # plot the function
    x = np.linspace(np.min(x), np.max(x), num_points)
    return ax.plot(x, eval(func, x, *args), **kwargs)

def find_ratio(func1:Union[str,'function'], args1:np.ndarray, func2:Union[str,'function'], args2:np.ndarray, pct_1:float, x:Union[tuple,np.ndarray]):
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
    '''
    # get the range
    x_min, x_max = np.min(x), np.max(x)
    
    # define the function to minimize
    def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple) -> float:
        return np.abs((1-pct_1)*eval(func1, x_, *args1_) - pct_1*eval(func2, x_, *args2_))
    
    # minimize it!
    res = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),))[0]
    
    # print a warning if the edge is at a boundary
    if (abs(x-x_min) < 1e-3) or (abs(x-x_max) < 1e-3):
        print('WARNING: analysis.find_ratio picked a value at the edge of a range.')

    return res

def find_value(func:Union[str,'function'], args:tuple, target:float, x:np.ndarray):
    ''' Find where the function equals a target value.

    Parameters
    ----------
    func : Union[str,function]
        Function to find the value of.
    args : tuple
        Additional arguments for the function.
    target : float
        Target value to hit.
    x : np.ndarray
        X values, the search will be limited within the minimum and maximum of this range.
    '''
    # get bounds
    x_min, x_max = np.min(x), np.max(x)
    
    # define the function to minimize
    def min_me(x_, args_) -> float:
        return np.abs(eval(func, x_, *args_) - target)
    
    # minimize it
    res = opt.brute(min_me, args=(args,), ranges=((x_min, x_max),))[0]

    # print a warning if x is at an edge
    if (abs(res-x_min) < 1e-3) or (abs(res-x_max) < 1e-3):
        print('WARNING: analysis.find_ratio picked a value at the edge of a range.')
    
    return res