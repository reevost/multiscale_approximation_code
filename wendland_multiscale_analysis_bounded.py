import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import halton_points
import wendland_functions
import time
plot_flag = True
# Original paper: Multiscale analysis in Sobolev spaces on bounded domains - Holger Wendland
tic_start = time.perf_counter()


def data_multilevel_structure(data, n=4, mu=0.5, starting_mesh_norm=None, starting_data_point=None):
    # data is the set of data points from which we want to build our nested sequence of data sets.
    # n is the number of level on which we want to split our data.
    # mu is a value in (0,1) which describe the relation between the mesh norm of two nested data sets: h(X_j+1) = mu * h(X_j)
    # starting_mesh_norm is the mesh norm used to find the first set of data.

    if starting_mesh_norm is None:
        mesh_norm = np.min([np.max(data, axis=0)[0]-np.min(data, axis=0)[0], np.max(data, axis=0)[1]-np.min(data, axis=0)[1]])/2
    else:
        mesh_norm = 2*starting_mesh_norm  # for the code purpose it is better to consider the twice of the mesh norm

    if starting_data_point is None:
        starting_data_point = data[0]

    data_nest_list = []  # initialize the empty array where we will put the nested sets

    # find the new set based on the starting mesh norm
    new_data = []  # future set with the actual mesh norm
    selected_point = starting_data_point
    temp_data = data
    while temp_data.size:
        # filter the list of points removing the point that are too close with the selected point / selected started point
        temp_data = temp_data[np.sum((selected_point - temp_data)**2, axis=1) > mesh_norm**2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

        # add the point to the set of point with the actual mesh norm
        new_data += [selected_point]

        # If the filtered list in non-empty chose a new point to reiterate the process. as new point the closest to the oldest one will be chosen
        if temp_data.size:
            selected_point = min(temp_data, key=lambda p: np.sum((p - temp_data[0]) ** 2))

    # save the filtered set with the actual mesh norm in the list of nested sets
    data_nest_list += [np.array(new_data)]
    print("The number of points in the level 1 is ", len(new_data))

    for j in np.arange(1, n):
        # update the mesh norm
        mesh_norm *= mu

        temp_data = data
        # filter the list of points removing the point that are too close with the points already in the set before
        for point in data_nest_list[j-1]:
            temp_data = temp_data[np.sum((point - temp_data) ** 2, axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

        if temp_data.size:
            selected_point = temp_data[0]

            while temp_data.size:
                # filter the list of points removing the point that are too close with the selected point / selected started point
                temp_data = temp_data[np.sum((selected_point - temp_data) ** 2, axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

                # add the point to the set of point with the actual mesh norm
                new_data += [selected_point]

                # If the filtered list in non-empty chose a new point to reiterate the process. as new point the closest to the oldest one will be chosen
                if temp_data.size:
                    selected_point = min(temp_data, key=lambda p: np.sum((p - temp_data[0]) ** 2))
        else:
            print("full at level", j)
        # save the filtered set with the actual mesh norm in the list of nested sets
        data_nest_list += [np.array(new_data)]
        print("The number of points in the level", j+1, "is ", len(new_data))

    return data_nest_list


# =====================================================================================================================================================================================
# =========================================== START OF THE CODE =======================================================================================================================
# =====================================================================================================================================================================================
# domain
domain_x = np.linspace(0, 1, 100)  # INPUT
domain_y = np.linspace(0, 1, 100)  # INPUT
domain_meshed_x, domain_meshed_y = np.meshgrid(domain_x, domain_y)

# true function
true_function = lambda lx, ly: np.exp(-lx ** 4) * lx + np.exp(-ly ** 4) * ly  # INPUT
true_f_in_domain = true_function(domain_meshed_x, domain_meshed_y)

# given the samples on the domain we build the nested sequence of sets
number_of_Levels = 4  # INPUT
x, y = halton_points.halton_sequence(200, 2)  # INPUT - data sites
sampled_points = np.concatenate((np.array([x]).T, np.array([y]).T), axis=1)
d = sampled_points.shape[1]+1
mesh_norm_0, mu_coefficient = np.min([np.max(sampled_points, axis=0)[0]-np.min(sampled_points, axis=0)[0], np.max(sampled_points, axis=0)[1]-np.min(sampled_points, axis=0)[1]])/8, 0.5  # INPUT
nest = data_multilevel_structure(sampled_points, n=number_of_Levels, starting_mesh_norm=mesh_norm_0, mu=mu_coefficient)
gamma = 1  # INPUT
nu_coefficient = 1/mesh_norm_0  # INPUT in this case nu has the highest possible value
# nu_coefficient = gamma/mu_coefficient  # INPUT in this case nu has the lowest possible value
print("evaluating the nest of sets and all the parameters in", time.perf_counter()-tic_start)

# nest plot
color_map = plt.get_cmap("viridis")
if plot_flag:
    for i in np.arange(number_of_Levels):
        plt.figure(figsize=[7, 7])
        plt.scatter(nest[i][:, 0], nest[i][:, 1], color=color_map(np.linspace(0, .8, number_of_Levels))[i])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title("subset $X_%d$" % (i+1))
        plt.show()

solving_technique = ["block matrix", "single level iteration"][1]

if solving_technique == "block matrix":
    # initializing the rhs and the block matrix
    rhs_f_list = []
    interpolation_block_matrix = np.array([])

    tic = time.perf_counter()  # time counter

    for level in np.arange(number_of_Levels):  # routine starting - solving the problem building the triangular block matrix
        sub_level_set = nest[level]
        delta_j = mesh_norm_0 * nu_coefficient * mu_coefficient ** level  # scaling parameter of the compactly supported rbf

        # add to the list of function rhs the values of the actual set
        rhs_f_list += [np.array([[true_function(p[0], p[1]) for p in sub_level_set]]).T]

        # evaluate A_j and B_jk for k < j.
        # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
        A_j = sparse.csr_matrix([[(delta_j**-d)*wendland_functions.wendland_function(np.linalg.norm(a-b)/delta_j) for b in sub_level_set] for a in sub_level_set])

        if level == 0:
            # in level == 0 there aren't the matrices B_jk
            interpolation_block_matrix = A_j
        else:
            # in level != 0 there are the matrices B_jk
            B_jk_list = []
            for k in np.arange(level):
                delta_k = delta_j/(mu_coefficient ** (level-k))
                B_jk_list += [np.array([[(delta_k ** -d) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_k) for b in nest[k]] for a in sub_level_set])]
            B_jk = sparse.csr_matrix(np.concatenate(B_jk_list, axis=1))

            interpolation_block_matrix = sparse.csr_matrix(np.block([
                [interpolation_block_matrix.toarray(), sparse.bsr_matrix((B_jk.shape[1], B_jk.shape[0]), dtype=np.int8).toarray()],
                [B_jk.toarray(), A_j.toarray()]
            ]))
        print("ended level", level+1)


    # evaluate the column of the right hand side f in the points of the subset sequence until the current subset
    rhs_f = np.concatenate(rhs_f_list)

    # solve find the list of coefficient alpha_j of the approximant at the level j
    alpha_full_vector = scipy.sparse.linalg.cg(interpolation_block_matrix, rhs_f)
    print("time needed to build and solve the system: ", time.perf_counter()-tic, "seconds")

    function_approximation, error_approximation, cumulative_number_of_points = [0*domain_meshed_x], [true_f_in_domain], 0
    for level in np.arange(number_of_Levels):  # routine for the evaluation on the domain of the interpolants at the different levels
        tic = time.perf_counter()
        sub_level_set = nest[level]
        number_of_points_at_level_j = len(sub_level_set)
        delta_j = mesh_norm_0 * nu_coefficient * mu_coefficient ** level  # scaling parameter of the compactly supported rbf

        # recovering alpha_j
        alpha_j = alpha_full_vector[cumulative_number_of_points:cumulative_number_of_points+number_of_points_at_level_j]

        # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
        approximant_domain_j = np.array([[np.dot(np.array([[(delta_j**-d)*wendland_functions.wendland_function(np.linalg.norm(np.array([a, b]) - x_j)/delta_j) for x_j in sub_level_set]]), alpha_j)[0][0] for a in domain_x] for b in domain_y])

        cumulative_number_of_points += number_of_points_at_level_j  # in order to recover the correct alpha_j

        # update error_approximation and function_approximation
        function_approximation += [function_approximation[-1]+approximant_domain_j]
        error_approximation += [error_approximation[-1]-approximant_domain_j]
        print("time needed to evaluate the approximant", level+1, "on the domain: ", time.perf_counter() - tic, "seconds")

elif solving_technique == "single level iteration":
    # initializing the rhs and the block matrix
    tic = time.perf_counter()  # time counter

    function_approximation, error_approximation = [0 * domain_meshed_x], [true_f_in_domain]
    for level in np.arange(number_of_Levels):  # routine starting - solving the problem solving at every iteration the linear system
        sub_level_set = nest[level]
        delta_j = mesh_norm_0 * nu_coefficient * mu_coefficient ** level  # scaling parameter of the compactly supported rbf

        alpha_j = []
        rhs = [np.array([[true_function(p[0], p[1]) for p in sub_level_set]]).T]  # the initial rhs id just f evaluated on the points

        # evaluate A_j and B_jk for k < j.
        # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
        A_j = sparse.csr_matrix([[(delta_j ** -d) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_j) for b in sub_level_set] for a in sub_level_set])

        for k in np.arange(level):
            # in level != 0 there are the matrices B_jk
            delta_k = delta_j / (mu_coefficient ** (level - k))
            B_jk = sparse.csr_matrix(np.array([[(delta_k ** -d) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_k) for b in nest[k]] for a in sub_level_set]))
            rhs -= np.dot(B_jk, alpha_j[k])

        # solve the linear system at level j
        alpha_j += [scipy.sparse.linalg.cg(A_j, rhs)[0]]
        print("ended level", level + 1, "in", time.perf_counter()-tic, "seconds")

    # solve find the list of coefficient alpha_j of the approximant at the level j

    function_approximation, error_approximation = [0 * domain_meshed_x], [true_f_in_domain]
    for level in np.arange(number_of_Levels):  # routine for the evaluation on the domain of the interpolants at the different levels
        tic = time.perf_counter()
        sub_level_set = nest[level]
        number_of_points_at_level_j = len(sub_level_set)
        delta_j = mesh_norm_0 * nu_coefficient * mu_coefficient ** level  # scaling parameter of the compactly supported rbf

        # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
        approximant_domain_j = np.array([[np.dot(alpha_j[level], np.array([[(delta_j ** -d) * wendland_functions.wendland_function(np.linalg.norm(np.array([a, b]) - x_j) / delta_j) for x_j in sub_level_set]]))[0][0] for a in domain_x] for b in domain_y])

        # update error_approximation and function_approximation
        function_approximation += [function_approximation[-1] + approximant_domain_j]
        error_approximation += [error_approximation[-1] - approximant_domain_j]
        print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic,
              "seconds")
# elif solving_technique == "sparse linear operator":


# plot of the true function / approximation function / approximation error
if plot_flag:
    fig = plt.figure(figsize=[25, 10])
    for i in range(number_of_Levels+1):
        ax = fig.add_subplot(3, number_of_Levels+1, 1+i, projection='3d')
        ax.plot_surface(domain_meshed_x, domain_meshed_y, true_f_in_domain, cmap=cm.Spectral, linewidth=0, antialiased=False)
        ax.set_title("true function", fontsize="small")

        ax = fig.add_subplot(3, number_of_Levels+1, 2+number_of_Levels+i, projection='3d')
        ax.plot_surface(domain_meshed_x, domain_meshed_y, function_approximation[i], cmap=cm.Spectral, linewidth=0, antialiased=False)
        ax.set_title("approximation function at step %d" % i, fontsize="small")

        ax = fig.add_subplot(3, number_of_Levels+1, 3+2*number_of_Levels+i, projection='3d')
        ax.plot_surface(domain_meshed_x, domain_meshed_y, error_approximation[i], cmap=cm.Spectral, linewidth=0, antialiased=False)
        ax.set_title("approximation error at step %d" % i, fontsize="small")
    plt.show()
