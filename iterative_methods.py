import numpy as np
import wendland_functions
from scipy import sparse
import time


class IterativeCounter(object):
    def __init__(self, disp=True, input_type="residual"):
        self._disp = disp
        self.input_type = input_type
        self.niter = 0
        self.exp = 0
        self.iterlist = []

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp & (self.input_type == "residual"):
            # print('iter %3i\trk = %s' % (self.niter, str(rk)))
            while rk < 10 ** -self.exp:
                self.exp += 1
                self.iterlist += [self.niter]
        else:  # cg setting, i.e. we have a np.array to deal with
            self.iterlist = self.niter  # just save the last value


def jacobi(A, b, eps=10 ** -8, x_0=None, callback=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x_0 is None:
        x = np.zeros(len(A[0]))
    else:
        if len(x_0.shape) == 2:
            x = x_0.reshape((-1,))
        else:
            x = x_0
    if len(b.shape) == 2:
        b = b.reshape((-1, ))
    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for until the residual is smaller than eps
    r = 1
    while r > eps:
        x = (b - np.dot(R, x)) / D
        r = np.linalg.norm(b - np.dot(A, x))
        if callback is not None:
            callback(r)
    return x


def matrix_multiscale_approximation(nested_set, right_hand_side, h_list, nu, wendland_coefficients=(1, 3),
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
    dim = domain[0].shape[1]
    # c_dim = domain[1].shape[1]  # possible future extension
    function_approximation_list, error_approximation_list = [np.zeros((domain[0].shape[0], 1))], [
        domain[1].reshape((domain[0].shape[0], -1))]  # set the initial values for f and e
    number_of_levels_ = len(nested_set)
    tolerance = 10 ** -8  # tolerance for the solver
    print("solving technique:", solving_technique)
    if solving_technique in ["cg", "gmres"]:
        # initializing the rhs and the block matrix
        rhs_f_list = []
        interpolation_block_matrix = np.array([])
        tic = time.perf_counter()

        # routine starting - solving the problem building the triangular block matrix
        for level in np.arange(number_of_levels_):
            sub_level_set = nested_set[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6) - knowing that will be expensive
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf

            # add to the list of function rhs the values of the actual set
            rhs_f_list += [np.array(
                [right_hand_side[:, -1:][~np.abs(right_hand_side[:, :-1] - tmp_p).any(axis=1)][0] for tmp_p in
                 sub_level_set])]  # should be parallelized

            # evaluate A_level and B_jk for k=level.
            # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
            A_j = sparse.csr_matrix([[(delta_j ** dim) *
                                      wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_j,
                                                                           k=wendland_coefficients[0],
                                                                           d=wendland_coefficients[1]) for b in
                                      sub_level_set] for a in sub_level_set])  # should be parallelized

            # the list of all matrix on column "level" on the big triangular block matrix. i.e. A_level and B_jk with fixed k=level and variable j=level, ..., number_of_levels.
            column_level_list = [A_j]
            for j in np.arange(level + 1, number_of_levels_):
                # here we add B_jk for K=level, and j=level+1, ..., number of levels
                column_level_list += [np.array([[(delta_j ** -dim) *
                                                 wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_j,
                                                                                      k=wendland_coefficients[0],
                                                                                      d=wendland_coefficients[1]) for
                                                 b in sub_level_set] for a in nested_set[j]])]  # should be parallelized
            column_level = sparse.vstack(column_level_list)

            if level == 0:
                interpolation_block_matrix = column_level
            else:
                # add a block of zeros over the other matrices with sparse.bsr_matrix((row_length, column_length), dtype=np.int8).toarray()
                column_level = sparse.vstack([sparse.bsr_matrix(
                    (sum([len(nested_set[jj]) for jj in np.arange(level)]), len(nested_set[level])),
                    dtype=np.int8).toarray(), column_level])
                # concatenate the columns
                interpolation_block_matrix = sparse.hstack([interpolation_block_matrix, column_level])
            print("built column", level + 1)

        # evaluate the column of the right-hand side f in the points of the subset sequence until the current subset
        rhs_f = np.concatenate(rhs_f_list)
        # solve find the list of coefficient alpha_j of the approximant at the level j
        if solving_technique == "gmres":
            iter_counter = IterativeCounter()
            # noinspection PyUnresolvedReferences
            alpha_full_vector, iter_full_system = scipy.sparse.linalg.gmres(interpolation_block_matrix, rhs_f, tol=tolerance, callback=iter_counter)
        elif solving_technique == "cg":
            iter_counter = IterativeCounter(input_type="x")
            # noinspection PyUnresolvedReferences
            alpha_full_vector, iter_full_system = scipy.sparse.linalg.cg(
                interpolation_block_matrix.T @ interpolation_block_matrix, interpolation_block_matrix.T @ rhs_f,
                tol=tolerance, callback=iter_counter)
        else:
            raise Exception("Sorry, invalid solving technique inserted")
        print("time needed to build and solve the system: ", time.perf_counter() - tic, "seconds")
        print("iteration needed for converge with tol equal to 10**-position:", iter_counter.iterlist)

        cumulative_number_of_points = 0
        # routine for the evaluation on the domain of the interpolants at the different levels
        for level in np.arange(number_of_levels_):
            tic = time.perf_counter()
            sub_level_set = nested_set[level]
            number_of_points_at_level_j = len(sub_level_set)
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf

            # recovering alpha_j
            alpha_j = alpha_full_vector[cumulative_number_of_points:cumulative_number_of_points + number_of_points_at_level_j]

            # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
            approximant_domain_j = np.array([[sparse.csr_matrix(np.array([[(delta_j ** -dim) *
                                                                           wendland_functions.wendland_function(np.linalg.norm(p_ - x_j) / delta_j,
                                                                                                                k=wendland_coefficients[0],
                                                                                                                d=wendland_coefficients[1])
                                                                           for x_j in sub_level_set]])).dot(alpha_j)[0]
                                              for p_ in domain[0]]]).T
            cumulative_number_of_points += number_of_points_at_level_j  # in order to recover the correct alpha_j
            # update error_approximation and function_approximation
            function_approximation_list += [function_approximation_list[-1] + approximant_domain_j]
            error_approximation_list += [error_approximation_list[-1] - approximant_domain_j]
            print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic, "seconds")
    elif solving_technique == "multistep_iteration":
        # initializing the rhs and the block matrix
        alpha_list = []
        iter_list = []
        tic = time.perf_counter()

        for level in np.arange(number_of_levels_):  # routine starting - solving the problem at level "level"
            sub_level_set = nested_set[level]
            # the list of function rhs is the values of the actual set
            rhs_f_level = np.array([right_hand_side[:, -1:][~np.abs(right_hand_side[:, :-1] - tmp_p).any(axis=1)][0] for tmp_p in sub_level_set])  # should be parallelized

            # the compactly supported rbf chosen is delta_j^-d * Phi(||x-y||/delta_j), where Phi is a wendland function
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6) - knowing that will be expensive
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf
            # print("delta", delta_j)
            # evaluate A_level.
            A_j = sparse.csr_matrix([[(delta_j ** -dim) * wendland_functions.wendland_function(
                np.linalg.norm(a - b) / delta_j, k=wendland_coefficients[0], d=wendland_coefficients[1]) for b in
                                      sub_level_set] for a in sub_level_set])  # should be parallelized
            # print(A_j)
            # print("rhs\n", np.array([right_hand_side[:, :][~np.abs(right_hand_side[:, :-1] - tmp_p).any(axis=1)][0] for tmp_p in sub_level_set]))
            # update the rhs = f- sum_k<level B_level,k alpha_k.
            for k in np.arange(level):
                delta_k = nu * h_list[k]
                B_lk = sparse.csr_matrix([[(delta_k ** -dim) * wendland_functions.wendland_function(np.linalg.norm(a - b) / delta_k,
                                                                                        k=wendland_coefficients[0],
                                                                                        d=wendland_coefficients[1]) for
                                           b in nested_set[k]] for a in sub_level_set])  # should be parallelized
                rhs_f_level -= B_lk.dot(alpha_list[k].reshape(len(nested_set[k]), 1))  # alpha_k has shape (N_k,) when here is needed (N_k, 1).

            # print("rhs_updated\n", rhs_f_level)
            # solve find the list of coefficient alpha_j of the approximant at the level j
            iter_counter = IterativeCounter(input_type="x")
            # noinspection PyUnresolvedReferences
            alpha_val, iter_val = scipy.sparse.linalg.cg(A_j, rhs_f_level, tol=tolerance, callback=iter_counter)
            print("list of iteration needed for converge of step", level + 1, "with tol", tolerance, ":", iter_counter.iterlist)
            alpha_list += [alpha_val]
            iter_list += [iter_val]
            # print("alpha", level+1, ":\n", np.array([alpha_list[level]]).T)

        print("time needed to build and solve the system: ", time.perf_counter() - tic, "seconds")

        # routine for the evaluation on the domain of the interpolants at the different levels
        for level in np.arange(number_of_levels_):
            tic = time.perf_counter()
            sub_level_set = nested_set[level]
            # mesh_norm_j = halton_points.fill_distance(sub_level_set, 10**-6)  # TO DO LIST
            # delta_j = mesh_norm_j * nu  # scaling parameter of the compactly supported rbf
            delta_j = nu * h_list[level]  # scaling parameter of the compactly supported rbf

            # compute the approximant at the step j: s_j(x) = [Phi_j(x, x_j)] @ alpha_j
            approximant_domain_j = np.array([[sparse.csr_matrix(np.array([[(delta_j ** -dim) *
                                                                           wendland_functions.wendland_function(np.linalg.norm(p_ - x_j) / delta_j,
                                                                                        k=wendland_coefficients[0],
                                                                                        d=wendland_coefficients[1])
                                                                           for x_j in sub_level_set]])).dot(alpha_list[level])[0]
                                              for p_ in domain[0]]]).T
            # update error_approximation and function_approximation
            function_approximation_list += [function_approximation_list[-1] + approximant_domain_j]
            error_approximation_list += [error_approximation_list[-1] - approximant_domain_j]
            print("time needed to evaluate the approximant", level + 1, "on the domain: ", time.perf_counter() - tic, "seconds")
    return function_approximation_list, error_approximation_list
