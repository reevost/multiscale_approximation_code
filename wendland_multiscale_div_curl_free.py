import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as integrate

import halton_points
import wendland_functions
import time
import os

# Original paper: Multiscale analysis in Sobolev spaces on bounded domains - Holger Wendland
plot_flag = True
save = False
tic_start = time.perf_counter()
cwd = os.getcwd()  # get the working directory


class IterativeCounter(object):
    def __init__(self, disp=True, solver="gmres"):
        self._disp = disp
        self.solver = solver
        self.niter = 0
        self.exp = 0
        self.iterlist = []

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp & (self.solver == "gmres"):
            # print('iter %3i\trk = %s' % (self.niter, str(rk)))
            while rk < 10 ** -self.exp:
                self.exp += 1
                self.iterlist += [self.niter]
        else:  # cg setting, i.e. we have a np.array to deal with
            self.iterlist = self.niter  # just save the last value


def data_multilevel_structure(data, number_of_levels_=4, mu=0.5, starting_mesh_norm=None, starting_data_point=None,
                              nest_flag=True):
    # data is the set of data points from which we want to build our nested sequence of data sets.
    # number_of_levels is the number of level on which we want to split our data.
    # mu is a value in (0,1) which describe the relation between the mesh norm of two nested data sets: h(X_j+1) = mu * h(X_j)
    # starting_mesh_norm is the mesh norm used to find the first set of data.

    if starting_mesh_norm is None:
        mesh_norm = np.min(
            [np.max(data, axis=0)[0] - np.min(data, axis=0)[0], np.max(data, axis=0)[1] - np.min(data, axis=0)[1]]) / 4
    else:
        mesh_norm = starting_mesh_norm

    if starting_data_point is None:
        starting_data_point = data[0]

    data_nest_list = []  # initialize the empty array where we will put the nested sets

    # find the new set based on the starting mesh norm
    new_data = []  # future set with the actual mesh normTrue
    selected_point = starting_data_point
    temp_data = data
    while temp_data.size:
        # filter the list of points removing the point that are too close with the selected point / selected started point
        temp_data = temp_data[np.sum((selected_point - temp_data) ** 2,
                                     axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

        # add the point to the set of point with the actual mesh norm
        new_data += [selected_point]

        # If the filtered list in non-empty chose a new point to reiterate the process. as new point the closest to the oldest one will be chosen
        if temp_data.size:
            selected_point = min(temp_data, key=lambda temp_p: np.sum((temp_p - temp_data[0]) ** 2))

    # save the filtered set with the actual mesh norm in the list of nested sets
    data_nest_list += [np.array(new_data)]
    print("The number of points in the level 1 is", len(new_data))

    for j_ in np.arange(1, number_of_levels_):
        # update the mesh norm
        mesh_norm *= mu
        # filter the list of points removing the point that are too close with the points already in the set before
        for point in data_nest_list[j_ - 1]:
            temp_data = temp_data[np.sum((point - temp_data) ** 2,
                                         axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2
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
                temp_data = temp_data[np.sum((selected_point - temp_data) ** 2,
                                             axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

                # add the point to the set of point with the actual mesh norm
                new_data += [selected_point]

                # If the filtered list in non-empty chose a new point to reiterate the process. as new point the closest to the oldest one will be chosen
                if temp_data.size:
                    selected_point = min(temp_data, key=lambda temp_p: np.sum((temp_p - temp_data[0]) ** 2))
        else:
            print("full at level", j_)
        # save the filtered set with the actual mesh norm in the list of nested sets
        data_nest_list += [np.array(new_data)]
        print("The number of points in the level", j_ + 1, "is", len(new_data))

    return data_nest_list


# def approximation_via_kernel()

# =================================================================================================================================================================================================================================================================================
# =========================================== START OF THE CODE ===================================================================================================================================================================================================================
# =================================================================================================================================================================================================================================================================================
d = 3  # dimension.
points_on_each_axis = 101  # INPUT
number_of_levels = 3  # INPUT
full_points_number = 10000

# given the samples on the domain we build the nested sequence of sets, and eventually save them to a file
# scattered data points
theta, phi = np.array(halton_points.halton_sequence(full_points_number, 2))  # INPUT - data sites
x = np.cos(theta*2*np.pi)*np.sin(phi*np.pi)
y = np.sin(theta*2*np.pi)*np.sin(phi*np.pi)
z = np.cos(phi*np.pi)
sampled_points = np.concatenate((np.array([x]).T, np.array([y]).T, np.array([z]).T), axis=1)


# example 1 - Divergence-free RBFs on Surfaces, Francis J. Narcowich Joseph D. Ward
t_field = []
def rho_arg(phi_, theta_, phi_c): return np.sin(phi_c) * np.sin(phi_) + np.cos(phi_c) * np.cos(phi_) * np.cos(theta_)


for i in np.arange(full_points_number):
    def field_helper(phi_, theta_, phi_c): return np.arccos(rho_arg(phi_, theta_, phi_c))*np.exp(-(8*np.arccos(rho_arg(phi_, theta_, phi_c)))**2)/(np.sqrt(1-rho_arg(phi_, theta_, phi_c)**2))
    gamma_field_i = 2*8**2*np.cos(np.pi/4)*np.sin(theta[i])*(field_helper(phi[i], theta[i], np.pi/4)-field_helper(phi[i], theta[i], -np.pi/4))
    delta_field_i = 2*8**2*(field_helper(phi[i], theta[i], np.pi/4)*(np.cos(phi[i]*np.sin(np.pi/4))-np.sin(phi[i])*np.cos(np.pi/4)*np.cos(theta[i]))-field_helper(phi[i], theta[i], -np.pi/4)*(-np.cos(phi[i]*np.sin(np.pi/4))-np.sin(phi[i])*np.cos(np.pi/4)*np.cos(theta[i])))
    u_vector = np.array([gamma_field_i, delta_field_i])
    Q_i = np.array([[-np.sin(phi[i])*np.cos(theta[i]), -np.sin(theta[i])], [-np.sin(phi[i])*np.sin(theta[i]), np.cos(theta[i])], [np.cos(phi[i]), 0]])
    t_field += [Q_i.dot(u_vector)]
rhs = np.concatenate((sampled_points, np.array(t_field)), axis=1)

mesh_norm_0, mu_coefficient = np.min([np.max(sampled_points, axis=0)[0] - np.min(sampled_points, axis=0)[0],
                                      np.max(sampled_points, axis=0)[1] - np.min(sampled_points, axis=0)[
                                          1]]) / 4, 0.5  # INPUT
gamma = 1  # INPUT

nest = data_multilevel_structure(sampled_points, number_of_levels_=number_of_levels, starting_mesh_norm=0.5,
                                 mu=mu_coefficient, nest_flag=True)
mesh_norm_list = [halton_points.fill_distance(nest[0], sampled_points) * mu_coefficient ** level for level in
                  range(number_of_levels)]
nu_coefficient = gamma/mu_coefficient  # INPUT
print("evaluating the nest of sets and all the parameters in", time.perf_counter() - tic_start)


# nest plot
color_map = plt.get_cmap("viridis")
if plot_flag:
    for i in np.arange(number_of_levels):
        fig = plt.figure(i)
        ax = fig.add_subplot(projection="3d")
        ax.scatter(nest[i][:, 0], nest[i][:, 1], nest[i][:, 2], color=color_map(np.linspace(0, .8, number_of_levels+1))[i+1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.title("subset $X_%d$" % (i + 1))
        plt.show()


def matrix_multiscale_approximation(nested_set, right_hand_side, h_list, nu, wendland_coefficients=(3, 3),
                                    solving_technique="gmres", domain=None):
    # Function that makes the interpolation using the multiscale approximation technique (see holger paper)
    # PARAMETERS ========================================================================================================================
    # domain[0] : ndarray. Domain where the interpolator is evaluated. If domain is None, instead of evaluating the functions on the domain we store every interpolator as function
    # This is a key parameter for the function, since will change its behaviour.
    # We could have a slower but low memory function where nothing is stored and everything is computed when needed, or
    # we could have a classic function where we store into temporal variables the values, and we use them to have a simpler function.
    # domain[1] : ndarray. the evaluation of the original function on the domain points, crucial for plotting the error of the approximation process with respect to the levels.
    #
    # solving technique: "cg", "gmres" or "single_iteration". The first solve the system A@A.T x = b A.T using the conjugate gradient,
    # the second solve A x = b using gmres, while in both A = the big triangular matrix. The last one solve with cg iteratively the linear systems Ax=b level by level
    #
    # wendland_coefficients: tuple. A couple of integers (k, d) associated with the wanted Wendland compactly supported radial basis function phi_(d, k).
    #
    # nu: float. The coefficient that define the radius (delta) of the compactly supported rbf, delta_j = nu * mu_j, where mu_j is the mesh norm at the level j.
    #
    # right_hand_side: a ndarray, of shape (N, d+1). The blocks of the array with shapes (N,d) and (N,1) are respectively, the positions and the values of the right-hand side.
    # Namely, the first array has the position x_1, ..., x_N, ...  while the second has the values of the unknown function in that points f(x_1), ..., f(x_N), ...
    # Here, f: Omega subset R**d ---> Omega subset R. Of course is needed that the points of the nested set are included in the positions.
    #
    # h_list: list with length equal to the nested set length. h_list[i] contains the i-th fill_distance value related to the i-th set in nest. Provided as input just to speed up the process
    #
    # nested set: list of ndarray. The list of the nested sequence of sets X_1, ..., X_n.
    # RETURNS ============================================================================================================================
    # nested_approximations: list. Is a list of ndarray if the domain is given, otherwise is a list of functions.

    # At a certain point the controls for the parameters types and shapes should be included.

    # set the initial values for f and e, just used to store the evolution of the interpolation and the error. Used for representation
    error_approximation_list = [domain[1].reshape((domain[0].shape[0], -1))]
    function_approximation_list = [np.zeros(error_approximation_list[0].shape)]
    number_of_levels_ = len(nested_set)
    dim = domain[0].shape[1]
    tolerance = 10 ** -8  # tolerance for the solver
    print("solving technique:", solving_technique)
    if solving_technique in ["cg", "gmres"]:
        # initializing the rhs and the block matrix
        rhs_f_list = []
        interpolation_block_matrix = np.array([])
        tic = time.perf_counter()

        for level in np.arange(number_of_levels_):  # routine starting - solving the problem building the triangular block matrix
            sub_level_set = nested_set[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6) - knowing that will be expensive
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf

            # add to the list of function rhs the values of the actual set
            rhs_f_list += [np.array([right_hand_side[:, -dim:][~np.abs(right_hand_side[:, :-dim] - tmp_p).any(axis=1)][0] for tmp_p in sub_level_set]).reshape((-1, 1))]  # should be parallelized

            # evaluate A_level and B_jk for k=level.
            # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
            A_j = sparse.csr_matrix(np.block([[(delta_j ** -dim) * wendland_functions.div_wendland(a-b, c=delta_j, d=wendland_coefficients[0], k=wendland_coefficients[1], dim=dim) for b in sub_level_set] for a in sub_level_set]))  # should be parallelized

            column_level_list = [
                A_j]  # the list of all matrix on column "level" on the big triangular block matrix. i.e. A_level and B_jk with fixed k=level and variable j=level, ..., number_of_levels.
            for j in np.arange(level + 1, number_of_levels_):
                # here we add B_jk for K=level, and j=level+1, ..., number of levels
                column_level_list += [np.block([[(delta_j ** -dim) * wendland_functions.div_wendland(a-b, c=delta_j, d=wendland_coefficients[0], k=wendland_coefficients[1], dim=dim) for
                                                 b in sub_level_set] for a in nested_set[j]])]  # should be parallelized
            column_level = sparse.vstack(column_level_list)

            if level == 0:
                interpolation_block_matrix = column_level
            else:
                # add a block of zeros over the other matrices with sparse.bsr_matrix((row_length, column_length), dtype=np.int8).toarray()
                column_level = sparse.vstack([sparse.bsr_matrix(
                    (dim*sum([len(nested_set[jj]) for jj in np.arange(level)]), dim*len(nested_set[level])),
                    dtype=np.int8).toarray(), column_level])
                # concatenate the columns
                interpolation_block_matrix = sparse.hstack([interpolation_block_matrix, column_level])
            print("built column", level + 1)

        # evaluate the column of the right-hand side f in the points of the subset sequence until the current subset
        rhs_f = np.vstack(rhs_f_list)
        # solve find the list of coefficient alpha_j of the approximant at the level j
        if solving_technique == "gmres":
            iter_counter = IterativeCounter()
            # noinspection PyUnresolvedReferences
            alpha_full_vector, iter_full_system = scipy.sparse.linalg.gmres(interpolation_block_matrix, rhs_f,
                                                                            tol=tolerance, callback=iter_counter)
        elif solving_technique == "cg":
            iter_counter = IterativeCounter(solver="cg")
            # noinspection PyUnresolvedReferences
            alpha_full_vector, iter_full_system = scipy.sparse.linalg.cg(
                interpolation_block_matrix.T @ interpolation_block_matrix, interpolation_block_matrix.T @ rhs_f,
                tol=tolerance, callback=iter_counter)
        else:
            raise Exception("Sorry, invalid solving technique inserted")
        print("time needed to build and solve the system: ", time.perf_counter() - tic, "seconds")
        print("iteration needed for converge with tol equal to 10**-position:", iter_counter.iterlist)
        cumulative_number_of_points = 0
        for level in np.arange(number_of_levels_):  # routine for the evaluation on the domain of the interpolants at the different levels
            tic = time.perf_counter()
            sub_level_set = nested_set[level]
            number_of_points_at_level_j = dim*len(sub_level_set)
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf

            # recovering alpha_j
            alpha_j = alpha_full_vector[cumulative_number_of_points:cumulative_number_of_points + number_of_points_at_level_j]

            # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
            approximant_domain_j = np.array([sparse.csr_matrix(np.block([[(delta_j ** -dim) * wendland_functions.div_wendland(p_-x_j, c=delta_j, d=wendland_coefficients[0], k=wendland_coefficients[1], dim=dim) for x_j in sub_level_set]])).dot(alpha_j) for p_ in domain[0]])
            cumulative_number_of_points += number_of_points_at_level_j  # in order to recover the correct alpha_j
            # update error_approximation and function_approximation
            function_approximation_list += [function_approximation_list[-1] + approximant_domain_j]
            error_approximation_list += [error_approximation_list[-1] - approximant_domain_j]
            print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic,
                  "seconds")
    elif solving_technique == "multistep_iteration":
        # initializing the rhs and the block matrix
        alpha_list = []
        iter_list = []
        tic = time.perf_counter()

        for level in np.arange(number_of_levels_):  # routine starting - solving the problem at level "level"
            sub_level_set = nested_set[level]
            # the list of function rhs is the values of the actual set
            rhs_f_level = np.array([right_hand_side[:, -dim:][~np.abs(right_hand_side[:, :-dim] - tmp_p).any(axis=1)][0] for tmp_p in sub_level_set]).reshape((-1, 1))  # should be parallelized

            # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6) - knowing that will be expensive
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf
            # print("delta", delta_j)
            # evaluate A_level.
            A_j = sparse.csr_matrix(np.block([[(delta_j ** -dim) * wendland_functions.div_wendland(a-b, c=delta_j, d=wendland_coefficients[0], k=wendland_coefficients[1], dim=dim) for b in sub_level_set] for a in sub_level_set]))  # should be parallelized

            # print("rhs\n", np.array([right_hand_side[:, :][~np.abs(right_hand_side[:, :-1] - tmp_p).any(axis=1)][0] for tmp_p in sub_level_set]))
            # update the rhs = f- sum_k<level B_level,k alpha_k.
            for k in np.arange(level):
                delta_k = nu * h_list[k]
                B_lk = sparse.csr_matrix(np.block([[(delta_k ** -dim) * wendland_functions.div_wendland(a - b, c=delta_k, d=wendland_coefficients[0], k=wendland_coefficients[1], dim=dim) for b in nested_set[k]] for a in sub_level_set]))  # should be parallelized
                rhs_f_level -= B_lk.dot(alpha_list[k].reshape(dim*len(nested_set[k]), 1))  # alpha_k has shape (N_k,) when here is needed (N_k, 1).

            # print("rhs_updated\n", rhs_f_level)
            # solve find the list of coefficient alpha_j of the approximant at the level j
            iter_counter = IterativeCounter(solver="cg")
            # noinspection PyUnresolvedReferences
            alpha_val, iter_val = scipy.sparse.linalg.cg(A_j, rhs_f_level, tol=tolerance, callback=iter_counter)
            print("list of iteration needed for converge of step", level+1, "with tol", tolerance, ":",
                  iter_counter.iterlist)
            alpha_list += [alpha_val]
            iter_list += [iter_val]

        print("time needed to build and solve the system: ", time.perf_counter() - tic, "seconds")
        for level in np.arange(number_of_levels_):  # routine for the evaluation on the domain of the interpolants at the different levels
            tic = time.perf_counter()
            sub_level_set = nested_set[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf

            # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
            approximant_domain_j = np.array([sparse.csr_matrix(np.block([[(delta_j ** -dim) * wendland_functions.div_wendland(p_-x_j, c=delta_j, d=wendland_coefficients[0], k=wendland_coefficients[1], dim=dim) for x_j in sub_level_set]])).dot(alpha_list[level]) for p_ in domain[0]])
            # update error_approximation and function_approximation
            function_approximation_list += [function_approximation_list[-1] + approximant_domain_j]
            error_approximation_list += [error_approximation_list[-1] - approximant_domain_j]
            print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic,
                  "seconds")
    return function_approximation_list, error_approximation_list


# imported_nest = []
# for i in range(number_of_levels):
#     temp_arr = []
#     with open(cwd + "/levels/%d_level.csv" % (i+1), 'r') as csvfile:
#         # creating a csv reader object
#         csvreader = csv.reader(csvfile)
#         # extracting each data row one by one
#         for row in csvreader:
#             temp_arr.append([np.float64(i) for i in row])
#     imported_nest.append(np.array(temp_arr))

chosen_solving_technique = "multistep_iteration"  # out of "gmres", "cg", "multistep_iteration"
function_approximation, error_approximation = matrix_multiscale_approximation(nested_set=nest, right_hand_side=rhs, solving_technique=chosen_solving_technique, h_list=mesh_norm_list, nu=nu_coefficient, domain=(sampled_points, np.array(t_field)))

# plot of the true function / approximation function / approximation error
if plot_flag:
    fig = plt.figure(figsize=[25, 15])
    for i in range(number_of_levels + 1):
        ax = fig.add_subplot(3, number_of_levels + 1, 1 + i, projection='3d')
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], linewidth=0, antialiased=False)
        ax.set_title("original function", fontsize="small")

        ax = fig.add_subplot(3, number_of_levels + 1, 2 + number_of_levels + i, projection='3d')
        ax.scatter(function_approximation[i][:, 0], function_approximation[i][:, 1], function_approximation[i][:, 2], color=color_map(np.linspace(0, .8, number_of_levels+1))[i], linewidth=0, antialiased=False)
        ax.set_title("approximation function at step %d" % i, fontsize="small")

        ax = fig.add_subplot(3, number_of_levels + 1, 3 + 2 * number_of_levels + i, projection='3d')
        ax.scatter(error_approximation[i][:, 0], error_approximation[i][:, 1], error_approximation[i][:, 2], color=color_map(np.linspace(0, .8, number_of_levels+1))[i], linewidth=0, antialiased=False)
        ax.set_title("approximation error at step %d" % i, fontsize="small")
    if save:
        plt.savefig(cwd + "/images/%d/multiscale_approximation_nu2_%s.png" % (number_of_levels, chosen_solving_technique),
                    transparent=False)
    plt.show()
