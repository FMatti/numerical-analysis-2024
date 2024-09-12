import numpy as np
import numpy as np


def bisection(fun, a, b, tol, nmax):
    """
    Find a zero of a scalar continuous function using the bisection method.

    Parameters
    ----------
    fun : callable
        The function for which to find the zero.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    tol : float
        The desired tolerance for the zero.
    nmax : int
        The maximum number of iterations.

    Returns
    -------
    zero : float
        The estimated zero of the function.
    res : float
        The value of the residual at the estimated zero.
    niter : int
        The iteration number at which the zero was computed.
    inc : list of float
        The absolute differences between successive approximations (increments).
    """

    fa = fun(a)
    fb = fun(b)

    # check that we can apply bisection, and that the a,b are not the zero we
    # are looking for
    if fa * fb > 0:
        print("The sign of FUN at the extrema of the interval must be different")
        return
    elif fa == 0:
        zero = a
        res = 0
        niter = 0
        return zero, res, niter, [], 0
    elif fb == 0:
        zero = b
        res = 0
        niter = 0
        return zero, res, niter, [], 0

    # initialize loop variables
    niter = 0
    inc = (
        []
    )  # this one stores the increment, i.e. how much we move from one guess to the next one
    I = (b - a) / 2  # this is the interval semilength, i.e. the error estimate
    m = (a + b) / 2  # this is the guess for the zero

    while (I > tol) and (niter < nmax):

        niter = niter + 1
        m_old = m  # this will store the old guess (needed to compute increments)
        fm = fun(m)

        if fa * fm < 0:  # the zero is between a and m => swap b with m
            b = m
            fb = fm
            I = (b - a) / 2
            m = (a + b) / 2

        elif fm * fb < 0:  # the zero is between m and b => swap a with m
            a = m
            fa = fm
            I = (b - a) / 2
            m = (a + b) / 2

        else:  # fm==0 => no need to update m
            I = 0

        # compute the increment, i.e how much we moved from the previous guess
        # m_old
        inc.append(abs(m - m_old))  # inc=[inc abs(m-m_old)];

        if niter >= nmax and I > tol:
            print(
                "bisection stopped without converging to the desired tolerance "
                "because the maximum number of iterations was reached\n"
            )

    zero = m
    res = abs(fun(m))
    err = I

    return zero, res, niter, inc, err


def fixed_point(phi, x0, tol, nmax):
    """
    Fixed point iterations.

    Parameters
    ----------
    phi : callable
        The function representing the fixed point iteration.
    x0 : float
        The initial guess for the fixed point.
    tol : float
        The desired tolerance for the fixed point.
    nmax : int
        The maximum number of iterations.

    Returns
    -------
    x_seq : array-like
        The successive values of the fixed point iterations.
    res : array-like
        The value of the residual at each iteration.
    niter : int
        The number of iterations performed.
    """

    niter = 0
    x_seq = []
    x_seq.append(x0)

    xt = phi(x0)
    res = []
    res.append(x0 - xt)  # this measures ``how much x0 is far from the fixed point''

    while (abs(res[-1]) > tol) and (niter < nmax):
        niter = niter + 1
        x_seq.append(xt)
        x0 = xt
        xt = phi(xt)
        res.append(abs(x0 - xt))

    if niter >= nmax:
        print(
            [
                "fixedPoint stopped without converging to the desired\n "
                "tolerance because the maximum number of iterations was reached\n"
            ]
        )

    # convert from list to array
    x_seq = np.array(x_seq)
    res = np.array(res)

    return x_seq, res, niter


def newton(f, df, x0, tol, nmax, *kwargs):
    """
    Find a zero of a scalar continuous function using the Newton method.

    Parameters
    ----------
    fun : callable
        The function for which to find the zero.
    dfun : callable
        The derivative of the function.
    x0 : float
        The initial guess for the zero.
    tol : float
        The desired tolerance for the zero.
    nmax : int
        The maximum number of iterations.

    Returns
    -------
    zero : float
        The estimated zero of the function.
    res : float
        The value of the residual at the estimated zero.
    niter : int
        The iteration number at which the zero was computed.
    inc : list of float
        The absolute differences between successive approximations (increments).
    """

    x = x0
    fx = f(x, *kwargs)
    dfx = df(x, *kwargs)
    niter = 0
    inc = []
    diff = tol + 1

    while (diff >= tol) and (niter <= nmax):
        niter = niter + 1
        diff = -fx / dfx
        x = x + diff
        diff = abs(diff)

        fx = f(x, *kwargs)
        dfx = df(x, *kwargs)

        inc.append(diff)

    if niter > nmax:
        print(
            [
                "newton stopped without converging to the desired tolerance "
                + "because the maximum number of iterations was reached"
            ]
        )
    zero = x
    res = fx
    inc = np.array(inc)
    return zero, res, niter, inc


def newtonsys(F, J, x0, tol, nmax):
    """
    Find the zeros of a system of non-linear equations using the Newton method.

    Parameters
    ----------
    F : callable
        The system of continuous and differentiable functions.
    J : callable
        The function that returns the Jacobian matrix.
    x0 : float or array-like
        The initial guess for the zeros.
    tol : float
        The desired tolerance for convergence.
    nmax : int
        The maximum number of iterations.

    Returns
    -------
    xx : ndarray
        Array of intermediate solutions.
    inc : ndarray
        Array of increments between consecutive solutions.
    niter : int
        The number of iterations required for convergence.
    """

    x = x0
    niter = 0
    res = tol + 1
    xx = []
    xx.append(x0)
    inc = []

    while (np.linalg.norm(res) > tol) and (niter <= nmax):
        x = x - np.linalg.solve(J(x), F(x))
        res = F(x)
        niter = niter + 1
        xx.append(x)
        inc.append(np.linalg.norm(x - x0))
        x0 = x

    if niter > nmax:
        raise ValueError(
            "Newton stopped without converging to the desired tolerance "
            + "because the maximum number of iterations was reached"
        )

    xx = np.array(xx)
    inc = np.array(inc)
    return xx, inc, niter
