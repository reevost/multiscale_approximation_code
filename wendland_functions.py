import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as special
import scipy.integrate as integrate
plot_flag = False


def square_bracket_operator(alpha, grade):  # both integer numbers with alpha >= grade-1.
    if grade == -1:
        return 1/(alpha+1)
    elif grade == 0:
        return 1
    else:
        # return np.math.factorial(alpha)/np.math.factorial(alpha-grade)
        return np.prod(np.arange(alpha, alpha-grade, -1))


def curly_bracket_operator(nu, grade):  # both integer numbers.
    if grade == 0:
        return 1
    elif nu == 0:
        print("Probably error")
        return 0
    else:
        # return np.math.factorial(nu+grade-1)/np.math.factorial(nu-1)
        return np.prod(np.arange(nu + grade - 1, nu - 1, -1))


def beta(j, k_plus_one, nu):  # j <= k_plus_one, all parameters are positive integers.
    if j == 0 and k_plus_one == 0:
        return 1
    else:
        coefficient_sum, k = 0, k_plus_one-1
        for n in np.arange(max(j-1, 0), k_plus_one):
            coefficient_sum += beta(n, k, nu)*square_bracket_operator(n+1, n-j+1)/curly_bracket_operator(nu+2*k-n+1, n-j+2)
        return coefficient_sum


def wendland_function(r, d=3, k=1):
    # r "points" where the function is evaluated. r is supposed to be ||x-y|| i.e. the norm of some point difference, so r in R^+.
    # k is degree of the function, who is C^2k.
    # d is the dimension of the embedding space.
    if k == int(k):  # having odd d implies that the value of k is an arbitrary integer
        nu = int(d / 2) + k + 1  # optimal value for the purpose of the class function.
        progress_evaluation = 0
        for n in np.arange(0, k+1):
            progress_evaluation += beta(n, k, nu) * (r**n) * max(1-r, 0)**(nu+2*k+1)
        return progress_evaluation
    else:
        # we expect even dimension here, i.e. non integer k (associated to even values of d)
        nu = d/2 + k + 1/2
        if 1 > r >= 0:
            result = integrate.quad(lambda t: t*(1-t)**nu * (t**2 - r**2)**(k-1) / (special.gamma(k) * 2**(k-1)), r, 1)
            estimate_integration_value, upper_bound_error = result[0], result[1]
        else:
            estimate_integration_value = 0
        return estimate_integration_value


def curl_wendland(x_m_y, c, dim=3, d=3, k=3):
    return - np.eye(dim)*(wendland_function(np.linalg.norm(x_m_y/c), d=d+2, k=k-1))/c**2 \
           - np.outer(x_m_y, x_m_y)*(wendland_function(np.linalg.norm(x_m_y/c), d=d+4, k=k-2))/c**4


def div_wendland(x_m_y, c, dim=3, d=3, k=3):
    return np.eye(dim)*[(1-dim)*(wendland_function(np.linalg.norm(x_m_y/c), d=d+2, k=k-1))/c**2-np.linalg.norm(x_m_y)**2*(wendland_function(np.linalg.norm(x_m_y/c), d=d+4, k=k-2))/c**4] \
           + np.outer(x_m_y, x_m_y)*(wendland_function(np.linalg.norm(x_m_y/c), d=d+4, k=k-2))/c**4


if plot_flag:
    domain_x = np.linspace(-1, 1, 100)
    domain_y = np.linspace(-1, 1, 100)
    domain_meshed_x, domain_meshed_y = np.meshgrid(domain_x, domain_y)
    wend_in_domain = np.array([[wendland_function(np.linalg.norm([a, b]), d=3, k=1) for a in domain_x] for b in domain_y])
    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_axis_off()
    ax.set_title("Wendland function $\phi_{3,1}$", fontsize=18)
    ax.plot_surface(domain_meshed_x, domain_meshed_y, wend_in_domain, cmap=cm.Spectral, linewidth=0, antialiased=False)
    plt.show()
