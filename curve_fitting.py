import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def mean_square_error_fo(bin_values, lambda_: float, beta_: float) -> float:
    """Computes the mean square error of a first order
    linear model with parameters beta_ and lambda_, with
    data discretized in steps of 1 unit of time.
    
    Args:
        - bin_values: bin values of the engagement histogram
        - beta_: parameter modelling the initial response to
        the tweet. Should be positive
        - lambda_: decay constant which should take values
        less than 1
        
    Returns:
        - MSE: the mean square error
    """
    x_hat = beta_
    MSE = 0
    for x in bin_values:
        MSE += (x-x_hat)**2
        x_hat *= (1-lambda_)
    MSE = MSE / len(bin_values)
    return MSE


def eval_error(time, model_engagement, true_engagement) -> tuple[float, float, float]:
    """Computes the mean square error, and the ratio between the sum of
    squared residuals and the sum of the squared signal in continuous time
    
    Args:
        - time: time vector for the observations
        - model_engagement: predicted engagement of fitted model
        - true_engagement: engagement at times specified in time vector
        
    Returns:
        - MSE: mean square error
        - RSS_frac: ratio of residuals squared and sum_signal_squared
        - sum_signal_squared: sum of squared signal samples
    """
    res = model_engagement - true_engagement
    sq_residual = np.square(res)
    sq_signal = np.square(true_engagement)
    
    MSE = np.mean(sq_residual)
    sum_signal_squared = np.sum(sq_signal)
    RSS_frac = np.sum(sq_residual)/sum_signal_squared
    
    #plt.hist(res, bins=200)
    #plt.show()
    
    return MSE, RSS_frac, sum_signal_squared


def lognormal_MLE(x):
    mu = np.mean(np.log(x))
    sigma = np.mean((np.log(x)-mu)**2)
    return mu, sigma

def exponential(x_, lambda_, beta_):
    """Exponential decay with parameters lambda and beta."""
    return beta_*np.exp(-lambda_*x_)


def log_exponential(x_, lambda_, beta_):
    return np.log(exponential(x_, lambda_, beta_))


def log_exp(x_, lambda_, log_beta_):
    """Logarithm of exponential decay with parameters lambda and beta."""
    return log_beta_ - lambda_*x_


def power(x_, lambda_, beta_):
    """Power law."""
    return beta_*(np.power(np.array(x_, dtype=float), -lambda_))


def estimate_lin_decay(time, engagement):
    t, log_eng = [], []
    for i, e in enumerate(engagement):
        if e != 0:
            log_eng.append(np.log(e))
            t.append(time[i])
    
    time = np.array(t)
    log_eng = np.array(log_eng)
    
    lin_model = LinearRegression(fit_intercept=True)
    
    #log_eng = np.log(np.array(engagement)+1)
    lin_model.fit(time.reshape(-1, 1), log_eng.reshape(-1, 1))
    
    lambda_ = -lin_model.coef_[0][0]
    beta_ = np.exp(lin_model.intercept_[0])
    
    #print("coefs:", lambda_, beta_)
    
    return lambda_, beta_
    

def estimate_decay_parameters(time, engagement, loss_='linear', f_scale_=1.0):
    """Returns an estimate of the parameters of the
    exponential decay function of engagement over time.
    
    Args:
        - time: time vector
        - engagement: engagement at times specified
        in time vector
        
    Returns:
        - popt: optimal parameters
    """
    # lower and upper bounds on lambda and beta:
    # 1e-4 <= lambda <= 1e3, and 1 <= beta <= 1e9.
    bounds_ = ([1e-4, 1], [1e3, 1e9])
    init = [1, 1]
    
    # Optimization: Trust Region Reflective algorithm 'trf'
    # Levenberg-Marquardt ('lm') does not handle parameter bounds
    method_ = 'trf'
    
    # Linear fit (alt. L1 fit)
    popt, _ = curve_fit(exponential, time, engagement, p0=init,
                        bounds=bounds_, method=method_, loss=loss_, f_scale=f_scale_)
    
    # Log fit
    #engagement = np.array(engagement)
    #log_eng = np.log(engagement+1e-3)
    #bounds_ = ([1e-4, 0], [1e3, 9]) # 0 <= log(beta) <= 9
    #popt, _ = curve_fit(log_exponential, time, engagement, p0=init,
    #                    bounds=bounds_, method=method_)
    
    return popt


def delayed_biexponential(x_, alpha, beta, gamma, eta, sigma, delay):
    """Solution to second order system. Assumes that the time bins
    are spaced 1 hour apart (tau_d is the delay given in hours).
    
    Note that gamma and alpha are positive, whereas they are negative
    in the original equation system.
    """
    #tau_dn = int(tau_d*100)
    #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
    #return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 
    delay=int(delay)
    delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
    delayed[:delay] = 0
    return eta*np.exp(-alpha*x_) + delayed


def biexponential(x_, alpha, beta, gamma, rho):
    """Solution to system
    dx1/dt = alpha*x1(t) + beta*x2(t)
    dx2/dt = gamma*x2(t) + rho*d(t)
    """
    return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))


def u_biexponential(x_, alpha, beta, gamma, rho):
    """Solution to system
    dx1/dt = alpha*x1(t) + beta*x2(t)
    dx2/dt = gamma*x2(t) + rho*d(t)
    """
    if np.abs(gamma - alpha) < 1e-10:
        return np.exp(-alpha*(x_))*rho
    else:
        return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))


def estimate_unconstr_biexp(time, engagement):
    
    def biexponential_unconstr_opt(x_, alpha, beta, gamma, rho):
        """Solution to system
        dx1/dt = alpha*x1(t) + beta*x2(t)
        dx2/dt = gamma*x2(t) + rho*d(t)
        """
        if np.abs(gamma - alpha) < 1e-10:
            return np.exp(-alpha*(x_))*rho
        else:
            return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))
    
    method_ = 'lm'
    init = [1, 1, 1, 1]
    popt, _ = curve_fit(biexponential_unconstr_opt, time, engagement, p0=init, method=method_)
    a, b, g, r = popt[0], popt[1], popt[2], popt[3]
    return a, b, g, r


def estimate_biexponential(time, engagement, loss_='linear'):
    
    def biexponential_opt(x_, alpha, beta, d_gamma, rho):
        """Solution to system
        dx1/dt = alpha*x1(t) + beta*x2(t)
        dx2/dt = gamma*x2(t) + rho*d(t)
        """
        gamma = alpha + d_gamma
        return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))
    
    method_ = 'trf'
            # alpha, beta, d_gamma, rho
    bounds_ = ([1e-5, 1e-2, 1e-5, 1e-3],
               [1e3,  1e9,  1e3,  1e9])
    init = [1e-1, 1, 1e-1, 1]
    
    """Alt. bounds:
    bounds_ = ([0,    0,    1e-9, 1e-9],
               [1e9,  1e9,  1e9,  1e9])
    init = [1e-1, 1, 1e-1, 1]
    """
    
    popt, _ = curve_fit(biexponential_opt, time, engagement,
                        p0=init, bounds=bounds_, method=method_, loss=loss_)
    a, b, d_g, r = popt[0], popt[1], popt[2], popt[3]
    g = a + d_g
    return a, b, g, r


def estimate_second_order(time, engagement, delay, loss_='linear'):
    """Fit a second order model of type
    dx1/dt = ax1 + bx2 + nd(t) + sd(t-td)
    dx2/dt = gx2 +     + rd(t-td)
    Since r and b are coupled we set r=1
    
    Args:
        - time: time vector
        - engagement: engagement at times specified
        in time vector
        
    Returns:
        - a, b, g, n, s: model parameters
    """
    
    def biexponential_free_delay(x_, d_alpha, beta, gamma, eta, sigma, delay):
        """Solution to second order system. Assumes that the time bins
        are spaced 1 hour apart (tau_d is the delay given in hours).

        Note that gamma and alpha are positive, whereas they are negative
        in the original equation system.
        """
        #tau_dn = int(tau_d*100)
        #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
        #return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 
        delay_n=int(delay)
        alpha = gamma + d_alpha
        delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
        delayed[:delay_n] = 0
        return eta*np.exp(-alpha*x_) + delayed
    
    def biexponential_fixed_delay(x_, d_alpha, beta, gamma, eta, sigma):
        """Solution to second order system. Assumes that the time bins
        are spaced 1 hour apart (delay is given by encapsulating function hours).

        Note that gamma and alpha are positive, whereas they are negative
        in the original equation system.
        """
        #tau_dn = int(tau_d*100)
        #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
        alpha = gamma + d_alpha
        delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
        delayed[:delay] = 0
        return eta*np.exp(-alpha*x_) + delayed
        # return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 

    method_ = 'trf'
    if delay != 0:
        bounds_ = ([1e-5, 1e-2, 1e-5, 1,   1e-5],
                   [1e3, 1e9,   1e3,  1e9, 1e6])
        init = [1e-1, 1, 1e-1, 2, 1]
        popt, _ = curve_fit(biexponential_fixed_delay, time, engagement,
                            p0=init, bounds=bounds_, method=method_, loss=loss_)
        d_a, b, g, n, s, td = popt[0], popt[1], popt[2], popt[3], popt[4], 0
        
    else:
        tau_max = int(time[-1]-time[0])
        # instead of fitting gamma, use auxillary parameter c=alpha-gamma, which is
        # always less than 0 (if alpha and gamma are defined negative) and describes the
        # discrepancy between the types of decay. Using gamma < alpha < 0 will give an error.
        bounds_ = ([1e-5, 1e-2, 1e-5, 1, 1e-5, 0],
                   [1e3, 1e9,   1e3,  1e9, 1e6, tau_max])
        init = [1e-1, 1, 1e-1, 2, 1, int(0.5*tau_max)]

        # Linear fit (alt. L1 fit with 'soft_l1')
        popt, _ = curve_fit(biexponential_free_delay, time, engagement,
                            p0=init, bounds=bounds_, method=method_, loss=loss_)
        d_a, b, g, n, s, td = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    
    a = g + d_a
    return a, b, g, n, s, td
