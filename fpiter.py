import numpy as np

def fpiter(fun, x0, p = 2, tol = 1e-9, nmax = 1e5, verbose = True):
    """
        input:
        ------
            fun  : function (from Rd to Rd)
                   a continuous function, which governs the iteration process
            x0   : np.array (of shape d×1)
                   the initial value for the iteration process
            p    : 1, 2, np.Inf
                   determines the norm used to measure the error
            tol  : positive real
                   the error tolerance
            nmax : positive integer
                   maximum number of iterations

        output:
        -------
            xstar : real
                    the fixpoint
            hist  : array
                    iteration history
    """
    
    # initialize variables
    theta = 0
    newit = fun(x0)
    x = [x0]
    
    # debug
    if verbose == True:
        print("x" + str(len(x) - 1) + " = " + str(x[-1]))
    
    # iterate process until aposteriori error is below tol or number of iterations exceeds kmax
    while True:
        # update newit and theta
        x.append(newit)
        newit = fun(x[-1])
        theta = np.linalg.norm([newit - x[-1]], 2) / np.linalg.norm([x[-1] - x[-2]], 2)
        
        # debug
        if verbose == True:
            print("x" + str(len(x) - 1) + " = " + str(x[-1]) + " (theta = " + str(theta) + ")")
        
        # check if aposteriori error is below tol
        if theta < tol:
            x.append(newit)
            break
        # check if n > nmax
        if len(x) > nmax:
            break
        # check if convergence was established
        if theta > 1:
            break
    
    # process the result
    
    # debug
    if verbose == True:
        if theta > 1:
            print("warning! theta > 1 (theta = " + str(theta) + ")")
        if len(x) > nmax:
            print("warning! n exceeds nmax (n = " + str(len(x)) + ")")
        if len(x) <= nmax and theta < 1:
            print("x* = " + str(x[-1]))
    
    # return the result
    return {"xstar": x[-1], "hist": x}

