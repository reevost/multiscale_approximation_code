from typing import Callable, Any

import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import cm

import halton_points
import wendland_functions
import time

plot_flag = False
# Original paper: Multiscale analysis in Sobolev spaces on bounded domains - Holger Wendland
tic_start = time.perf_counter()


def data_multilevel_structure(data, number_of_levels=4, mu=0.5, starting_mesh_norm=None, starting_data_point=None, nest_flag=True):
    # data is the set of data points from which we want to build our nested sequence of data sets.
    # number_of_levels is the number of level on which we want to split our data.
    # mu is a value in (0,1) which describe the relation between the mesh norm of two nested data sets: h(X_j+1) = mu * h(X_j)
    # starting_mesh_norm is the mesh norm used to find the first set of data.

    if starting_mesh_norm is None:
        mesh_norm = np.min([np.max(data, axis=0)[0] - np.min(data, axis=0)[0], np.max(data, axis=0)[1] - np.min(data, axis=0)[1]]) / 2
    else:
        mesh_norm = 2 * starting_mesh_norm  # for the code purpose it is better to consider twice of the mesh norm

    if starting_data_point is None:
        starting_data_point = data[0]

    data_nest_list = []  # initialize the empty array where we will put the nested sets

    # find the new set based on the starting mesh norm
    new_data = []  # future set with the actual mesh norm
    selected_point = starting_data_point
    temp_data = data
    while temp_data.size:
        # filter the list of points removing the point that are too close with the selected point / selected started point
        temp_data = temp_data[np.sum((selected_point - temp_data) ** 2, axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

        # add the point to the set of point with the actual mesh norm
        new_data += [selected_point]

        # If the filtered list in non-empty chose a new point to reiterate the process. as new point the closest to the oldest one will be chosen
        if temp_data.size:
            selected_point = min(temp_data, key=lambda temp_p: np.sum((temp_p - temp_data[0]) ** 2))

    # save the filtered set with the actual mesh norm in the list of nested sets
    data_nest_list += [np.array(new_data)]
    print("The number of points in the level 1 is ", len(new_data))

    for j_ in np.arange(1, number_of_levels):
        # update the mesh norm
        mesh_norm *= mu
# filter the list of points removing the point that are too close with the points already in the set before
        for point in data_nest_list[j_ - 1]:
            temp_data = temp_data[np.sum((point - temp_data) ** 2, axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2
        temp_data = data
        if nest_flag:
            # filter the list of points removing the point that are too close with the points already in the set before
            for point in data_nest_list[j_ - 1]:
                temp_data = temp_data[np.sum((point - temp_data) ** 2,
                                             axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2
        else:
            new_data = []

        if temp_data.size:
            selected_point = temp_data[0]

            while temp_data.size:
                # filter the list of points removing the point that are too close with the selected point / selected started point
                temp_data = temp_data[np.sum((selected_point - temp_data) ** 2, axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

                # add the point to the set of point with the actual mesh norm
                new_data += [selected_point]

                # If the filtered list in non-empty chose a new point to reiterate the process. as new point the closest to the oldest one will be chosen
                if temp_data.size:
                    selected_point = min(temp_data, key=lambda temp_p: np.sum((temp_p - temp_data[0]) ** 2))
        else:
            print("full at level", j_)
        # save the filtered set with the actual mesh norm in the list of nested sets
        data_nest_list += [np.array(new_data)]
        print("The number of points in the level", j_ + 1, "is ", len(new_data))

    return data_nest_list


# def approximation_via_kernel()

# =====================================================================================================================================================================================
# =========================================== START OF THE CODE =======================================================================================================================
# =====================================================================================================================================================================================
# domain
domain_x = np.linspace(0, 1, 100)  # INPUT
domain_y = np.linspace(0, 1, 100)  # INPUT
domain_meshed_x, domain_meshed_y = np.meshgrid(domain_x, domain_y)

# true function
true_function: Callable[[Any, Any], Any] = lambda lx, ly: np.exp(-lx ** 4) * lx + np.exp(-ly ** 4) * ly  # INPUT
true_f_in_domain = true_function(domain_meshed_x, domain_meshed_y)

# given the samples on the domain we build the nested sequence of sets
number_of_levels_ = 4  # INPUT
x, y = halton_points.halton_sequence(700, 2)  # INPUT - data sites
sampled_points = np.concatenate((np.array([x]).T, np.array([y]).T), axis=1)

# create a hole in the samples
sampled_points_2 = []
for p in sampled_points:
    if np.linalg.norm(p-0.5) > 0.2:
        sampled_points_2 = sampled_points_2 + [p]
sampled_points = np.array(sampled_points_2)

d = sampled_points.shape[1] + 1
global mesh_norm_0, mu_coefficient
mesh_norm_0, mu_coefficient = np.min([np.max(sampled_points, axis=0)[0] - np.min(sampled_points, axis=0)[0], np.max(sampled_points, axis=0)[1] - np.min(sampled_points, axis=0)[1]]) / 8, 0.5  # INPUT

nest = data_multilevel_structure(sampled_points, number_of_levels=number_of_levels_, starting_mesh_norm=mesh_norm_0, mu=mu_coefficient, nest_flag=True)

gamma = 1  # INPUT
nu_coefficient = 1 / mesh_norm_0  # INPUT in this case nu has the highest possible value
# nu_coefficient = gamma/mu_coefficient  # INPUT in this case nu has the lowest possible value
print("evaluating the nest of sets and all the parameters in", time.perf_counter() - tic_start)

# nest plot
color_map = plt.get_cmap("viridis")
if plot_flag:
    for i in np.arange(number_of_levels_):
        plt.figure(figsize=[7, 7])
        plt.scatter(nest[i][:, 0], nest[i][:, 1], color=color_map(np.linspace(0, .8, number_of_levels_))[i])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title("subset $X_%d$" % (i + 1))
        plt.show()


def matrix_multiscale_approximation(nested_set, right_hand_side, nu, wendland_coefficients=(1, 3), solving_technique="gmres", domain=None):
    # Function that makes the interpolation using the multiscale approximation technique (see holger paper)
    # PARAMETERS ========================================================================================================================
    # domain : ndarray. Domain where the interpolator is evaluated. If domain is None, instead of evaluating the functions on the domain we store every interpolator as function
    # This is a key parameter for the function, since will change its behaviour.
    # We could have a slower but low memory function where nothing is stored and everything is computed when needed, or
    # we could have a classic function where we store into temporal variables the values, and we use them to have a simpler function.
    #
    # solving technique: "cg", "gmres" or "single_iteration". The first solve the system A@A.T x = b A.T using the conjugate gradient,
    # the second solve A x = b using gmres, while in both A = the big triangular matrix. The last one solve with cg iteratively the linear systems Ax=b level by level
    #
    # wendland_coefficients: tuple. A couple of integers (n, k) associated with the wanted Wendland compactly supported radial basis function phi_(n, k).
    #
    # nu: float. The coefficient that define the radius (delta) of the compactly supported rbf, delta_j = nu * mu_j, where mu_j is the mesh norm at the level j.
    #
    # right_hand_side: list of 2 ndarray, of shape (N, d) and (N, c) respectively. The arrays are, respectively the positions and the values of the right-hand side.
    # Namely, the first array has the position x_1, ..., x_N, ...  while the second has the values of the unknown function in that points f(x_1), ..., f(x_N), ...
    # Here, f: Omega subset R**d ---> Omega subset R**c. Of course is needed that the points of the nested set are included in the positions.
    #
    # nested set: list of ndarray. The list of the nested sequence of sets X_1, ..., X_n.
    # RETURNS ============================================================================================================================
    # nested_approximations: list. Is a list of ndarray if the domain is given, otherwise is a list of functions.

    # At a certain point the controls for the parameters types and shapes should be included.

    function_approximation, error_approximation = [0 * right_hand_side[1]], [right_hand_side[1]]  # set the initial values for f and e
    number_of_levels = len(nested_set)
    print("solving technique:", solving_technique)
    if solving_technique in ["cg", "gmres"]:
        # initializing the rhs and the block matrix
        rhs_f_list = []
        interpolation_block_matrix = np.array([])
        tic = time.perf_counter()
        global mesh_norm_0, mu_coefficient

        for level in np.arange(number_of_levels):  # routine starting - solving the problem building the triangular block matrix
            sub_level_set = nest[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * mesh_norm_0 * mu_coefficient ** level

            # add to the list of function rhs the values of the actual set
            rhs_f_list += [np.array([right_hand_side[1][~np.abs(right_hand_side[0]-tmp_p).any(axis=1)] for tmp_p in sub_level_set])]  # should be parallelized

            # evaluate A_level and B_jk for k=level.
            # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
            A_j = sparse.csr_matrix([[(delta_j ** -len(right_hand_side[0][0])) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for b in sub_level_set] for a in sub_level_set])  # should be parallelized

            column_level_list = [A_j]  # the list of all matrix on column "level" on the big triangular block matrix. i.e. A_level and B_jk with fixed k=level and variable j=level, ..., number_of_levels.
            for j in np.arange(level+1, number_of_levels):
                # here we add B_jk for K=level, and j=level+1, ..., number of levels
                column_level_list += [np.array([[(delta_j ** -len(right_hand_side[0][0])) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for b in sub_level_set] for a in nest[j]])]  # should be parallelized
            column_level = sparse.vstack(column_level_list)

            if level == 0:
                interpolation_block_matrix = column_level
            else:
                # add a block of zeros over the other matrices with sparse.bsr_matrix((row_length, column_length), dtype=np.int8).toarray()
                column_level = sparse.vstack([sparse.bsr_matrix((sum([len(nest[jj]) for jj in np.arange(level)]), len(nest[level])), dtype=np.int8).toarray(), column_level])
                # concatenate the columns
                interpolation_block_matrix = sparse.hstack([interpolation_block_matrix, column_level])
            print("built column", level + 1)

        # evaluate the column of the right-hand side f in the points of the subset sequence until the current subset
        rhs_f = np.concatenate(rhs_f_list)

        # solve find the list of coefficient alpha_j of the approximant at the level j
        if solving_technique == "gmres":
            # noinspection PyUnresolvedReferences
            alpha_full_vector = scipy.sparse.linalg.gmres(interpolation_block_matrix, rhs_f)[0]
        elif solving_technique == "cg":
            # noinspection PyUnresolvedReferences
            alpha_full_vector = scipy.sparse.linalg.cg(interpolation_block_matrix.T@interpolation_block_matrix, interpolation_block_matrix.T@rhs_f)[0]
        else:
            print("invalid solving technique inserted")
        print("time needed to build and solve the system: ", time.perf_counter() - tic, "seconds")
        cumulative_number_of_points = 0
        for level in np.arange(number_of_levels):  # routine for the evaluation on the domain of the interpolants at the different levels
            tic = time.perf_counter()
            sub_level_set = nest[level]
            number_of_points_at_level_j = len(sub_level_set)
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * mesh_norm_0 * mu_coefficient ** level  # scaling parameter of the compactly supported rbf

            # recovering alpha_j
            alpha_j = alpha_full_vector[cumulative_number_of_points:cumulative_number_of_points + number_of_points_at_level_j]

            # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
            approximant_domain_j = np.array([sparse.csr_matrix(np.array([[(delta_j ** -len(right_hand_side[0][0])) * wendland_functions.wendland_function(np.linalg.norm(p_ - x_j) / delta_j) for x_j in sub_level_set]])).dot(alpha_j)[0] for p_ in right_hand_side[0]])

            cumulative_number_of_points += number_of_points_at_level_j  # in order to recover the correct alpha_j

            # update error_approximation and function_approximation
            function_approximation += [function_approximation[-1] + approximant_domain_j]
            error_approximation += [error_approximation[-1] - approximant_domain_j]
            print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic, "seconds")
    elif solving_technique == "single_iteration":  # work in progress =======================================================================================================================================
        # initializing the rhs and the block matrix
        alpha_list = []
        interpolation_block_matrix = np.array([])
        tic = time.perf_counter()

        for level in np.arange(number_of_levels):  # routine starting - solving the problem at level "level"
            sub_level_set = nest[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * mesh_norm_0 * mu_coefficient ** level

            # the list of function rhs is the values of the actual set
            rhs_f_level = np.array([right_hand_side[1][~np.abs(right_hand_side[0] - tmp_p).any(axis=1)] for tmp_p in sub_level_set])  # should be parallelized

            # evaluate A_level.
            # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
            A_j = sparse.csr_matrix([[(delta_j ** -len(right_hand_side[0][0])) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for b in sub_level_set] for a in sub_level_set])  # should be parallelized

            # update the rhs = f- sum_k<level B_level,k alpha_k.
            delta_k = nu * mesh_norm_0
            for k in np.arange(level):
                delta_k *= mu_coefficient
                B_lk = sparse.csr_matrix([[(delta_k ** -len(right_hand_side[0][0])) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_k, k=wendland_coefficients[0], d=wendland_coefficients[1]) for b in sub_level_set] for a in sub_level_set])  # should be parallelized
                rhs_f_level -= B_lk@alpha_list[k]

            # solve find the list of coefficient alpha_j of the approximant at the level j
            # noinspection PyUnresolvedReferences
            alpha_list += [scipy.sparse.linalg.cg(A_j, rhs_f_level)[0]]

        print("time needed to build and solve the system: ", time.perf_counter() - tic, "seconds")
        for level in np.arange(number_of_levels):  # routine for the evaluation on the domain of the interpolants at the different levels
            tic = time.perf_counter()
            sub_level_set = nest[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * mesh_norm_0 * mu_coefficient ** level  # scaling parameter of the compactly supported rbf

            # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
            approximant_domain_j = np.array([sparse.csr_matrix(np.array([[(delta_j ** -len(right_hand_side[0][0])) * wendland_functions.wendland_function(np.linalg.norm(p_ - x_j) / delta_j) for x_j in sub_level_set]])).dot(alpha_list[level])[0] for p_ in right_hand_side[0]])

            # update error_approximation and function_approximation
            function_approximation += [function_approximation[-1] + approximant_domain_j]
            error_approximation += [error_approximation[-1] - approximant_domain_j]
            print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic,
                  "seconds")
    # return function_approximation, error_approximation
    return interpolation_block_matrix


rhs = [nest[-1], true_function(nest[-1][:, 0], nest[-1][:, 1])]
ma = matrix_multiscale_approximation(nested_set=nest, right_hand_side=rhs, nu=nu_coefficient)
print("COMPUTED VALUES")
ma_explicit, ncp = ma@np.eye(ma.shape[0]), 0
print("full matrix:", "\n||T||_2:", np.linalg.norm(ma_explicit, ord=2), "\n||T^-1||_2:", np.linalg.norm(np.linalg.inv(ma_explicit), ord=2))
eig_v, eig_mal, eig_mar = scipy.linalg.eig(ma_explicit, left=True, right=True)
svd_v = scipy.linalg.svdvals(ma_explicit)
eig_v = np.sort(eig_v)[::-1]
print("condition number of the change of basis matrix", np.linalg.cond(eig_mal@eig_mar))
plt.scatter(np.arange(len(eig_v)), eig_v, marker=".", c="b")
plt.scatter(np.arange(len(svd_v)), svd_v, marker=".", c="r")
plt.show()
print("min singular value:", svd_v[-1], "\nmax singular value:", svd_v[0])
for npp in [len(i) for i in nest]:
    ncp += npp
    ma_2 = ma_explicit[:ncp, :ncp]
    print(npp, ncp)
    print("||T||_2:", np.linalg.norm(ma_2, ord=2), "\n||T^-1||_2:", np.linalg.norm(np.linalg.inv(ma_2), ord=2))


# plot of the true function / approximation function / approximation error
# if plot_flag:
#    fig = plt.figure(figsize=[25, 10])
#     for i in range(number_of_levels_ + 1):
#         ax = fig.add_subplot(3, number_of_levels_ + 1, 1 + i, projection='3d')
#         ax.plot_surface(domain_meshed_x, domain_meshed_y, true_f_in_domain, cmap=cm.Spectral, linewidth=0,
#                         antialiased=False)
#         ax.set_title("true function", fontsize="small")
#
#         ax = fig.add_subplot(3, number_of_levels_ + 1, 2 + number_of_levels_ + i, projection='3d')
#         ax.plot_surface(domain_meshed_x, domain_meshed_y, function_approximation[i], cmap=cm.Spectral, linewidth=0,
#                         antialiased=False)
#         ax.set_title("approximation function at step %d" % i, fontsize="small")
#
#         ax = fig.add_subplot(3, number_of_levels_ + 1, 3 + 2 * number_of_levels_ + i, projection='3d')
#         ax.plot_surface(domain_meshed_x, domain_meshed_y, error_approximation[i], cmap=cm.Spectral, linewidth=0,
#                         antialiased=False)
#         ax.set_title("approximation error at step %d" % i, fontsize="small")
#     plt.show()
