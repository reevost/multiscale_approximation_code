from typing import Callable, Any

import numpy as np
import scipy
# from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import halton_points
import wendland_functions
import time

plot_flag = True
save = True
# Original paper: Multiscale analysis in Sobolev spaces on bounded domains - Holger Wendland
tic_start = time.perf_counter()
cwd = os.getcwd()  # get the working directory


def data_multilevel_structure(data, number_of_levels=4, mu=0.5, starting_mesh_norm=None, starting_data_point=None,
                              nest_=True):
    # data is the set of data points from which we want to build our nested sequence of data sets.
    # number_of_levels is the number of level on which we want to split our data.
    # mu is a value in (0,1) which describe the relation between the mesh norm of two nested data sets: h(X_j+1) = mu * h(X_j)
    # starting_mesh_norm is the mesh norm used to find the first set of data.

    if starting_mesh_norm is None:
        mesh_norm = np.min(
            [np.max(data, axis=0)[0] - np.min(data, axis=0)[0], np.max(data, axis=0)[1] - np.min(data, axis=0)[1]]) / 2
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
        temp_data = temp_data[np.sum((selected_point - temp_data) ** 2,
                                     axis=1) > mesh_norm ** 2]  # instead of sqrt(sum(square)) > value i consider sum(square) > value**2

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

        temp_data = data
        if nest_:
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
        print("The number of points in the level", j_ + 1, "is ", len(new_data))

    return data_nest_list


# =====================================================================================================================================================================================
# =========================================== START OF THE CODE =======================================================================================================================
# =====================================================================================================================================================================================
# domain
domain_x = np.linspace(0, 1, 100)  # INPUT
domain_y = np.linspace(0, 1, 100)  # INPUT
domain_meshed_x, domain_meshed_y = np.meshgrid(domain_x, domain_y)
d = 2


# true function
def true_function(*domain_points):
    def loc_f(lx): return np.exp(-lx ** 4) * lx  # INPUT
    sol = np.zeros(domain_points[0].shape)
    for i_axes in domain_points:
        sol += loc_f(i_axes)
    return sol


# given the samples on the domain we build the nested sequence of sets, and eventually save them to a file
number_of_levels_ = 6  # INPUT
if d == 2:
    x, y = halton_points.halton_sequence(10000, 2)  # INPUT - data sites
    sampled_points = np.concatenate((np.array([x]).T, np.array([y]).T), axis=1)
    rhs = true_function(np.array(x), np.array(y))
    # file = open("evaluated_data_2D.txt", "w")
    # for i in range(10000):
    #     file.write("%f,%f,%f\n" % (x[i], y[i], rhs[i]))
    # file.close()
else:  # we suppose to handle just dim = 2 or 3 for now.
    x, y, z = halton_points.halton_sequence(10000, 3)  # INPUT - data sites
    sampled_points = np.concatenate((np.array([x]).T, np.array([y]).T, np.array([z]).T), axis=1)
    rhs = true_function(np.array(x), np.array(y), np.array(z))
    # file = open("evaluated_data_3D.txt", "w")
    # for i in range(10000):
    #     file.write("%f,%f,%f,%f\n" % (x[i], y[i], z[i], rhs[i]))
    # file.close()


true_f_in_domain = true_function(domain_meshed_x, domain_meshed_y)
global mesh_norm_0, mu_coefficient
mesh_norm_0, mu_coefficient = np.min([np.max(sampled_points, axis=0)[0] - np.min(sampled_points, axis=0)[0],
                                      np.max(sampled_points, axis=0)[1] - np.min(sampled_points, axis=0)[
                                          1]]) / 8, 0.5  # INPUT

nest = data_multilevel_structure(sampled_points, number_of_levels=number_of_levels_, starting_mesh_norm=mesh_norm_0, mu=mu_coefficient, nest_=True)

gamma = 1  # INPUT
nu_coefficient = 1 / mesh_norm_0  # INPUT in this case nu has the highest possible value
# nu_coefficient = gamma/mu_coefficient  # INPUT in this case nu has the lowest possible value
print("evaluating the nest of sets and all the parameters in", time.perf_counter() - tic_start, "seconds")

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

A_v = []  # testing phase only


def cardinal_function(nested_set, nu, wendland_coefficients=(1, 3)):  # works only in nested case up to now
    # Function that makes the interpolation using the multiscale approximation technique (see holger paper)
    # PARAMETERS ========================================================================================================================
    # wendland_coefficients: tuple. A couple of integers (n, k) associated with the wanted Wendland compactly supported radial basis function phi_(n, k).
    #
    # nu: float. The coefficient that define the radius (delta) of the compactly supported rbf, delta_j = nu * mu_j, where mu_j is the mesh norm at the level j.
    #
    # nested set: list of ndarray. The list of the nested sequence of sets X_1, ..., X_n.
    # RETURNS ============================================================================================================================
    # cardinal_function: list. Is a list of ndarray if the domain is given, otherwise is a list of functions.

    # At a certain point the controls for the parameters types and shapes should be included.

    number_of_levels = len(nested_set)
    tic = time.perf_counter()
    finest_set = nest[-1]
    evaluated_cardinal_function = []
    global mesh_norm_0, mu_coefficient, A_v
    sub_level_set = nest[-1]

    for level in np.arange(number_of_levels - 1):
        sub_level_set = nest[level]
        # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
        # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf

        delta_j = nu * mesh_norm_0 * mu_coefficient ** level

        # evaluate A_level and B_nk for k=level.
        # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
        A_j = np.array([[(delta_j ** -len(sub_level_set[0])) * wendland_functions.wendland_function(
            np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for a in
                         sub_level_set] for b in sub_level_set])  # should be parallelized
        A_v += [A_j]
        B_nj = np.array([[(delta_j ** -len(sub_level_set[0])) * wendland_functions.wendland_function(
            np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for a in
                          finest_set] for b in sub_level_set])  # should be parallelized
        evaluated_cardinal_function += [(np.linalg.inv(A_j) @ B_nj).T]
        print("The time needed to evaluate the functions at step", level + 1, "is", time.perf_counter() - tic,
              "seconds")
    delta_j = nu * mesh_norm_0 * mu_coefficient ** (number_of_levels - 1)
    A_j = np.array([[(delta_j ** -len(sub_level_set[0])) * wendland_functions.wendland_function(
        np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for a in sub_level_set]
                    for b in sub_level_set])  # should be parallelized
    A_v += [A_j]
    return evaluated_cardinal_function


cf_list = cardinal_function(nested_set=nest, nu=nu_coefficient)
tau = 3 * 0.5 + 1 + 0.5  # wendland associated H^tau space tau = d/2 + k + 1/2, with wendland_function(k,d)
tic_2 = time.perf_counter()

chi_matrix = np.concatenate(tuple(cf_list + [np.eye(len(nest[-1]))]), axis=1)
for i_temp in np.arange(len(nest) - 1, 0, -1):
    temp_block = np.concatenate(tuple([cf[:len(nest[i_temp - 1]), :] for cf in cf_list[:i_temp]] + [
        np.zeros((len(nest[i_temp - 1]), sum([len(nest[i_temp_]) for i_temp_ in np.arange(i_temp, len(nest))])))]),
                                axis=1)
    chi_matrix = np.concatenate((temp_block, chi_matrix), axis=0)
# evaluate the inverse
chi_U, chi_S, chi_V = np.linalg.svd(chi_matrix)
inv_chi_matrix = chi_V.T@np.diag(1/chi_S)@chi_U.T
print("time to evaluate the inverse:", time.perf_counter() - tic_2, "seconds")
print("condition number of T@inv_T", np.linalg.cond(chi_matrix@inv_chi_matrix))
# test the 2-norm of the chi matrix and its inverse, to have an idea of the condition number of the matrix
print("np.linalg.norm(chi_matrix, ord=2)", np.linalg.norm(chi_matrix, ord=2))
print("np.linalg.norm(inv_chi_matrix, ord=2)", np.linalg.norm(inv_chi_matrix, ord=2))

# BUILD THE BLOCK FORMULATION TO EMPHASIZE NORM RESULTS
# initialize the matrices with zeros, and the dict of the rectangles for the image with the real size ratio of the blocks
norm_matrix, norm_matrix_bound = np.zeros((len(nest), len(nest))), np.zeros((len(nest), len(nest)))
norm_inv_matrix, norm_inv_matrix_bound = np.zeros((len(nest), len(nest))), np.zeros((len(nest), len(nest)))
rectangles, inv_rectangles = {}, {}
color_map = plt.get_cmap("YlOrRd")
# create the vector with the increasing size of the matrix with respect to "n"
cumulative_points = [0] + [sum([len(nest[temp]) for temp in np.arange(0, temp_2)]) for temp_2 in
                           np.arange(1, len(nest) + 1)]
# define a matrix where store in the position (k,j) the 2-norm of the respective block for every block under the diagonal. Do the same with the values of the theoretical bound. The same process is repeated for the inverse.
for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(i_temp + 1):
        # evaluate the norm of chi_ij and inv_chi_ij
        norm_temp = np.linalg.norm(chi_matrix[cumulative_points[i_temp]:cumulative_points[i_temp + 1],
                                   cumulative_points[j_temp]:cumulative_points[j_temp + 1]], ord=2)
        norm_inv_temp = np.linalg.norm(inv_chi_matrix[cumulative_points[i_temp]:cumulative_points[i_temp + 1],
                                       cumulative_points[j_temp]:cumulative_points[j_temp + 1]], ord=2)

        # store the values on the visual matrix np.linalg.norm(a - b) / delta_j
        norm_matrix[i_temp][j_temp] = norm_temp
        norm_inv_matrix[i_temp][j_temp] = norm_inv_temp
        if i_temp == j_temp:
            norm_matrix_bound[i_temp][j_temp] = 1
        else:
            norm_matrix_bound[i_temp][j_temp] = mu_coefficient ** (-d * (i_temp - j_temp) / 2)
        if i_temp - j_temp > 1:
            norm_inv_matrix_bound[i_temp][j_temp] = mu_coefficient ** (2 * tau * (i_temp - j_temp) - 2 * d * (i_temp+1))
        else:
            norm_inv_matrix_bound[i_temp][j_temp] = norm_matrix_bound[i_temp][j_temp]

# since we want to analyze the 1 and inf norm of the inverse we repeat the process above.
inv_norm_1, inv_norm_inf = np.zeros((len(nest), len(nest))), np.zeros((len(nest), len(nest)))
inv_norm_1_bound, inv_norm_inf_bound = np.zeros((len(nest), len(nest))), np.zeros((len(nest), len(nest)))

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(i_temp + 1):
        # evaluate the 1 and inf norm of inv_chi_ij
        norm_inv_1_temp = np.linalg.norm(inv_chi_matrix[cumulative_points[i_temp]:cumulative_points[i_temp + 1],
                                         cumulative_points[j_temp]:cumulative_points[j_temp + 1]], ord=1)
        norm_inv_inf_temp = np.linalg.norm(inv_chi_matrix[cumulative_points[i_temp]:cumulative_points[i_temp + 1],
                                           cumulative_points[j_temp]:cumulative_points[j_temp + 1]], ord=np.inf)
        # store the values on the visual matrices
        inv_norm_1[i_temp][j_temp] = norm_inv_1_temp
        inv_norm_inf[i_temp][j_temp] = norm_inv_inf_temp
        if i_temp == j_temp:
            inv_norm_1_bound[i_temp][j_temp] = 1
            inv_norm_inf_bound[i_temp][j_temp] = 1
        elif i_temp - j_temp >= 1:
            inv_norm_inf_bound[i_temp][j_temp] = mu_coefficient ** (tau * (i_temp - j_temp) - d * (i_temp+1) / 2)
            inv_norm_1_bound[i_temp][j_temp] = mu_coefficient ** (tau * (i_temp - j_temp) - 3 * d * (i_temp+1) / 2)
        else:
            inv_norm_1_bound[i_temp][j_temp] = mu_coefficient ** (-d * (i_temp - j_temp) / 2)
            inv_norm_inf_bound[i_temp][j_temp] = 1

# we need to split this in order to have a color map, i.e. to have a proper color map we need the max and min values, so we have to first compute all the values. and is not that expensive.
# so this routine can be incorporated above when I populate initially the matrices, since we do not have the max and the min values at the beginning, so we have to split this process afterwards. Indeed, the max value is used in the choice of color for every block.
for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(i_temp + 1):
        norm_temp = norm_matrix[i_temp][j_temp]
        norm_inv_temp = norm_inv_matrix[i_temp][j_temp]
        # build the blocks of the matrices above but with the real size ratio
        rectangles[(i_temp, j_temp)] = patches.Rectangle(
            (cumulative_points[j_temp], cumulative_points[-1] - cumulative_points[i_temp + 1]), len(nest[j_temp]),
            len(nest[i_temp]), color=color_map(norm_temp / np.max(norm_matrix)))
        inv_rectangles[(i_temp, j_temp)] = patches.Rectangle(
            (cumulative_points[j_temp], cumulative_points[-1] - cumulative_points[i_temp + 1]), len(nest[j_temp]),
            len(nest[i_temp]), color=color_map(norm_inv_temp / np.max(norm_inv_matrix)))

# define the plot figure - MATRICES NORM COMPARISON
fig = plt.figure(figsize=[15, 15])
ax = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax.title.set_text("chi matrix norm")
ax2.title.set_text("inv_chi matrix norm")
ax3.title.set_text("chi matrix norm bound")
ax4.title.set_text("inv_chi matrix norm bound")

# visualize the matrices as color maps with respect to the values
ax.matshow(norm_matrix, cmap=plt.cm.YlOrRd)
ax2.matshow(norm_inv_matrix, cmap=plt.cm.YlOrRd)
ax3.matshow(norm_matrix_bound, cmap=plt.cm.YlOrRd)
ax4.matshow(norm_inv_matrix_bound, cmap=plt.cm.YlOrRd)

# add the value of each block to the visualization
for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(norm_matrix[j_temp][i_temp], ".3f")
        ax.text(i_temp, j_temp, str(c), va="center", ha="center")

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(norm_inv_matrix[j_temp][i_temp], ".3f")
        ax2.text(i_temp, j_temp, str(c), va="center", ha="center")

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(norm_matrix_bound[j_temp][i_temp], ".3f")
        ax3.text(i_temp, j_temp, str(c), va="center", ha="center")

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(norm_inv_matrix_bound[j_temp][i_temp], ".3f")
        ax4.text(i_temp, j_temp, str(c), va="center", ha="center")

if save:
    plt.savefig(cwd + "/images/%d/chi_norm_matrix_comparison.png" % number_of_levels_, transparent=False)
plt.show()

# 1 and infinity norm bounds
fig_i1 = plt.figure(figsize=[15, 15])
ax = fig_i1.add_subplot(221)
ax2 = fig_i1.add_subplot(222)
ax3 = fig_i1.add_subplot(223)
ax4 = fig_i1.add_subplot(224)
ax.title.set_text("inv_chi matrix 1-norm")
ax2.title.set_text("inv_chi matrix inf-norm")
ax3.title.set_text("inv_chi matrix 1-norm bound")
ax4.title.set_text("inv_chi matrix inf-norm bound")

# visualize the matrices as color maps with respect to the values
ax.matshow(inv_norm_1, cmap=plt.cm.YlOrRd)
ax2.matshow(inv_norm_inf, cmap=plt.cm.YlOrRd)
ax3.matshow(inv_norm_1_bound, cmap=plt.cm.YlOrRd)
ax4.matshow(inv_norm_inf_bound, cmap=plt.cm.YlOrRd)

# add the value of each block to the visualization
for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(inv_norm_1[j_temp][i_temp], ".3e")
        ax.text(i_temp, j_temp, str(c), va="center", ha="center")

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(inv_norm_inf[j_temp][i_temp], ".3e")
        ax2.text(i_temp, j_temp, str(c), va="center", ha="center")

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(inv_norm_1_bound[j_temp][i_temp], ".3e")
        ax3.text(i_temp, j_temp, str(c), va="center", ha="center")

for i_temp in np.arange(len(nest)):
    for j_temp in np.arange(len(nest)):
        c = format(inv_norm_inf_bound[j_temp][i_temp], ".3e")
        ax4.text(i_temp, j_temp, str(c), va="center", ha="center")

if save:
    plt.savefig(cwd + "/images/%d/inv_chi_norm_matrix_1_inf_comparison.png" % number_of_levels_, transparent=False)
plt.show()

if plot_flag:  # other plots - matrix size and singular values
    # define the plot figure - MATRICES SIZE VISUALIZATION
    fig_b = plt.figure(figsize=[15, 7])
    ax3b = fig_b.add_subplot(121)
    ax3b.set_xlim((0, cumulative_points[-1]))
    ax3b.set_ylim((0, cumulative_points[-1]))
    ax3b.set_aspect("equal")
    ax4b = fig_b.add_subplot(122)
    ax4b.set_xlim((0, cumulative_points[-1]))
    ax4b.set_ylim((0, cumulative_points[-1]))
    ax4b.set_aspect("equal")

    for r in rectangles:
        ax3b.add_artist(rectangles[r])
        # the following instructions should be used to print the values inside the rectangles but with different sizes isn't really useful since most of them are not readable
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width() / 2.0
        cy = ry + rectangles[r].get_height() / 2.0
        # ax3.annotate(format(norm_matrix[r[0], r[1]], ".2f"), (cx, cy), color="k", weight="bold", fontsize=6, ha="center", va="center")

    for r in inv_rectangles:
        ax4b.add_artist(inv_rectangles[r])
        # the following instructions should be used to print the values inside the rectangles but with different sizes isn't really useful since most of them are not readable
        rx, ry = inv_rectangles[r].get_xy()
        cx = rx + inv_rectangles[r].get_width() / 2.0
        cy = ry + inv_rectangles[r].get_height() / 2.0
        # ax4.annotate(format(norm_inv_matrix[r[0], r[1]], ".2f"), (cx, cy), color="k", weight="bold", fontsize=6, ha="center", va="center")

    if save:
        plt.savefig(cwd + "/images/%d/chi_norm_matrix_size.png" % number_of_levels_, transparent=False)
    plt.show()

    # plots of the norm compared with theoretical results
    chi_matrix_norm_vector = np.array([np.linalg.norm(chi_matrix[:n, :n], ord=2) for n in cumulative_points[1:]])
    inv_chi_matrix_norm_vector = np.array(
        [np.linalg.norm(inv_chi_matrix[:n, :n], ord=2) for n in cumulative_points[1:]])
    cond_number_chi_matrix = chi_matrix_norm_vector * inv_chi_matrix_norm_vector

    # CHI MATRIX
    fig2 = plt.figure("chi matrix norm", figsize=[15, 15])
    ax = fig2.add_subplot(221)
    ax2 = fig2.add_subplot(222)
    ax.title.set_text("chi matrix norm")
    ax2.title.set_text("log plot")
    ax.plot(np.arange(1, number_of_levels_ + 1), chi_matrix_norm_vector, "b-", label="computed norm")
    ax.plot(np.arange(1, number_of_levels_ + 1),
            [mu_coefficient ** (-d * n / 2) for n in np.arange(1, number_of_levels_ + 1)], "r-",
            label="theoretical norm bound")
    ax.legend()
    ax.set_xlabel('number of steps')
    ax2.semilogy(np.arange(1, number_of_levels_ + 1), chi_matrix_norm_vector, "b--", label="computed norm")
    ax2.semilogy(np.arange(1, number_of_levels_ + 1),
                 [mu_coefficient ** (-d * n / 2) for n in np.arange(1, number_of_levels_ + 1)], "r--",
                 label="theoretical norm bound")
    ax2.legend()
    ax2.set_xlabel('number of steps')
    ax3 = fig2.add_subplot(223)
    ax4 = fig2.add_subplot(224)
    ax3.title.set_text("chi matrix norm")
    ax4.title.set_text("chi matrix norm bound")
    ax3.matshow(norm_matrix, cmap=plt.cm.YlOrRd)
    ax4.matshow(norm_matrix_bound, cmap=plt.cm.YlOrRd)

    # add the value of each block to the visualization
    for i_temp in np.arange(len(nest)):
        for j_temp in np.arange(len(nest)):
            c = format(norm_matrix[j_temp][i_temp], ".3f")
            ax3.text(i_temp, j_temp, str(c), va="center", ha="center")

    for i_temp in np.arange(len(nest)):
        for j_temp in np.arange(len(nest)):
            c = format(norm_matrix_bound[j_temp][i_temp], ".3f")
            ax4.text(i_temp, j_temp, str(c), va="center", ha="center")

    if save:
        plt.savefig(cwd + "/images/%d/chi_matrix_norm.png" % number_of_levels_, transparent=False)
    plt.show()

    # INVERSE CHI MATRIX
    fig3 = plt.figure("inverse chi matrix norm", figsize=[15, 15])
    ax = fig3.add_subplot(221)
    ax2 = fig3.add_subplot(222)
    ax.title.set_text("inverse chi matrix norm")
    ax2.title.set_text("log plot")
    ax.plot(np.arange(1, number_of_levels_ + 1), inv_chi_matrix_norm_vector, "b-", label="computed norm")
    ax.plot(np.arange(1, number_of_levels_ + 1),
            [mu_coefficient ** (-d * n) for n in np.arange(1, number_of_levels_ + 1)], "r-",
            label="theoretical norm bound")
    ax.legend()
    ax.set_xlabel('number of steps')
    ax2.semilogy(np.arange(1, number_of_levels_ + 1), inv_chi_matrix_norm_vector, "b--", label="computed norm")
    ax2.semilogy(np.arange(1, number_of_levels_ + 1),
                 [mu_coefficient ** (-n * d) for n in np.arange(1, number_of_levels_ + 1)], "r--",
                 label="theoretical norm bound")
    ax2.legend()
    ax2.set_xlabel('number of steps')
    ax3 = fig3.add_subplot(223)
    ax4 = fig3.add_subplot(224)
    ax3.title.set_text("inv chi matrix norm")
    ax4.title.set_text("inv chi matrix norm bound")
    ax3.matshow(norm_inv_matrix, cmap=plt.cm.YlOrRd)
    ax4.matshow(norm_inv_matrix_bound, cmap=plt.cm.YlOrRd)

    # add the value of each block to the visualization
    for i_temp in np.arange(len(nest)):
        for j_temp in np.arange(len(nest)):
            c = format(norm_inv_matrix[j_temp][i_temp], ".3f")
            ax3.text(i_temp, j_temp, str(c), va="center", ha="center")

    for i_temp in np.arange(len(nest)):
        for j_temp in np.arange(len(nest)):
            c = format(norm_inv_matrix_bound[j_temp][i_temp], ".3f")
            ax4.text(i_temp, j_temp, str(c), va="center", ha="center")
    if save:
        plt.savefig(cwd + "/images/%d/inverse_chi_matrix_norm.png" % number_of_levels_, transparent=False)
    plt.show()

    # CONDITION NUMBER
    fig4 = plt.figure("chi matrix condition number", figsize=[15, 7])
    ax = fig4.add_subplot(121)
    ax2 = fig4.add_subplot(122)
    ax.title.set_text("chi matrix condition number")
    ax2.title.set_text("log plot")
    ax.plot(np.arange(1, number_of_levels_ + 1), cond_number_chi_matrix, "b-", label="computed condition number")
    ax.plot(np.arange(1, number_of_levels_ + 1), [mu_coefficient ** (-d * n / 2 - 1) for n in np.arange(1, number_of_levels_ + 1)], "g-", label="aimed theoretical condition number bound")
    ax.plot(np.arange(1, number_of_levels_ + 1), [mu_coefficient ** (-d * n * (3 / 2)) for n in np.arange(1, number_of_levels_ + 1)], "r-", label="theoretical condition number bound")
    ax.legend()
    ax.set_xlabel('number of steps')
    ax2.semilogy(np.arange(1, number_of_levels_ + 1), cond_number_chi_matrix, "b--", label="computed condition number")
    ax2.semilogy(np.arange(1, number_of_levels_ + 1), [mu_coefficient ** (-d * n / 2 - 1) for n in np.arange(1, number_of_levels_ + 1)], "g--", label="aimed theoretical condition number bound")
    ax2.semilogy(np.arange(1, number_of_levels_ + 1), [mu_coefficient ** (-d * n * (3 / 2)) for n in np.arange(1, number_of_levels_ + 1)], "r--", label="theoretical condition number bound")
    ax2.legend()
    ax2.set_xlabel('number of steps')
    if save:
        plt.savefig(cwd + "/images/%d/condition_number_chi_matrix.png" % number_of_levels_, transparent=False)
    plt.show()

    # evaluate the singular values
    svd_values_vector = [scipy.linalg.svdvals(chi_matrix[:n, :n]) for n in cumulative_points[1:]]
    high_sv_count_1, high_sv_count_01, high_sv_count_10 = [], [], []

    # visualize the singular value at each level
    for n in np.arange(number_of_levels_):
        svd_v = svd_values_vector[n]
        high_sv_count_01 += [np.sum(svd_v > 0.1)]
        high_sv_count_1 += [np.sum(svd_v > 1)]
        high_sv_count_10 += [np.sum(svd_v > 10)]
        fig5 = plt.figure("singular values analysis", figsize=[7, 7])
        ax = fig5.add_subplot(111)
        ax.title.set_text("level " + str(n + 1))
        ax.scatter(np.arange(len(svd_v)), svd_v, marker=".", c="b")
        ax.plot(np.arange(len(svd_v)), np.ones(len(svd_v)) * 0.1, "r--")
        ax.plot(np.arange(len(svd_v)), np.ones(len(svd_v)), "r--")
        ax.plot(np.arange(len(svd_v)), np.ones(len(svd_v)) * 10, "r--")
        if save:
            plt.savefig(cwd + "/images/%d/singular_values_level_%d.png" % (number_of_levels_, n), transparent=False)
        plt.show()

    # plot the analysis on the high singular values
    fig6 = plt.figure("high singular values analysis", figsize=[15, 15])
    ax = fig6.add_subplot(221)
    ax2 = fig6.add_subplot(222)
    ax3 = fig6.add_subplot(223)
    ax4 = fig6.add_subplot(224)
    ax.plot(np.arange(1, number_of_levels_ + 1), [len(svd_n) for svd_n in svd_values_vector], "b-",
            label="total number of singular values"), ax.legend()
    ax2.plot(np.arange(1, number_of_levels_ + 1), high_sv_count_1, "b-", label="eps = 1"), ax2.legend()
    ax3.plot(np.arange(1, number_of_levels_ + 1), high_sv_count_01, "b-", label="eps = 0.1"), ax3.legend()
    ax4.plot(np.arange(1, number_of_levels_ + 1), high_sv_count_10, "b-", label="eps = 10"), ax4.legend()
    if save:
        plt.savefig(cwd + "/images/%d/number_of_singular_values_greater_than_eps.png" % number_of_levels_,
                    transparent=False)
    plt.show()
