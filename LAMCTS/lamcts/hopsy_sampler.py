import hopsy
import numpy as np
from scipy.optimize import linprog

######### old "black-box" likelihood approach #############
# class LAMCTS_hopsy_model:
#     def __init__(self, path,dim):
#         self.path = path
#         self.dim = dim
#     def compute_negative_log_likelihood(self,x):
#         path = self.path
#         global_constrs = path[0]
#         constr_satisfied = True
#         # we will add the inequality constraint using the hopsy function
#         # if global_constrs["A_ineq"] is not None and global_constrs["b_ineq"] is not None:
#         #    A = global_constrs["A_ineq"]
#         #    b = global_constrs["b_ineq"]
#         #    constr_satisfied = constr_satisfied and np.all(A@x <= b)
#         # could add branch for A_eq, but not necessary
#         for node in path[1:]: # skip global constraint
#             boundary = node[0].classifier.svm
#             x_reshape = x.reshape(1,-1)
#             constr_satisfied = constr_satisfied and boundary.predict(x_reshape) == node[1]
#             # node[1] store the direction to go
#         if constr_satisfied:
#             return -np.log(1)
#         else:
#             return np.inf
# 
# def propose_rand_samples_hopsy(num_samples, init_point, path, lb, ub, dim, thin=10, threads=12):
#     """
#     sample a function using hopsy?
#     """
#     #TODO: add exception for if region is empty
#     model = LAMCTS_hopsy_model(path, dim)
#     global_constrs = path[0]
#     constr_satisfied = True
#     # we will add the inequality constraint using the hopsy function
#     if global_constrs["A_ineq"] is not None and global_constrs["b_ineq"] is not None:
#         A = global_constrs["A_ineq"]
#         b = global_constrs["b_ineq"]
#     else: # no global constraint
#         A = np.array([np.ones(dim)])
#         b = np.array([np.inf])
#     problem = hopsy.Problem(A, b, model)
#     problem = hopsy.add_box_constraints(problem=problem,lower_bound=lb,upper_bound=ub)
# 
#     # try making one markov chain for each thread?
#     # mc = hopsy.MarkovChain(problem, 
#     #     proposal=hopsy.UniformCoordinateHitAndRunProposal, 
#     #     starting_point=init_point)
#     mc = [
#         hopsy.MarkovChain(
#             problem, proposal=hopsy.DikinWalkProposal, starting_point=init_point
#         )
#         for i in range(threads)
#     ]
#     # randomly generate the starting seed
#     rng = [hopsy.RandomNumberGenerator(seed=np.random.randint(100)) for i in range(threads)]
#     acceptance_rate, states = hopsy.sample(mc, rng, 
#         n_samples=num_samples, 
#         thinning=thin,
#         n_threads=threads)
#     print("hopsy sampler acceptance rate:",acceptance_rate)
#     # nested 2d array? idk
#     return acceptance_rate, states[0]


class LAMCTS_hopsy_model:
    # uniform distribution everywhere lmao
    # actually maybe you should make it over the hypercube lol
    def __init__(self, path,dim):
        self.path = path
        self.dim = dim
    def compute_negative_log_likelihood(self,x):
        return -np.log(1)

    
def find_cheb_center(A,b,lb,ub):
    """
    given a polytope Ax <= b and lb <= x <= ub, 
    find the center of the ball of maximal radius contained
    within the polytope
    """
    A_row_norms = np.linalg.norm(A,axis=1,ord=2)
    cheb_A = np.concatenate([A,np.array([A_row_norms]).T],axis=1)
    
    # -1 since scipy.linprog solves minimization problems
    cheb_c = np.zeros(np.shape(cheb_A)[1])
    np.put(a=cheb_c,ind=-1,v=-1)
    
    # setup upper/lower bounds
    ub_lb = list(zip(lb,ub))
    ub_lb.append((0,None))
    
    soln = linprog(
        c=cheb_c,
        A_ub=cheb_A,
        b_ub=b, 
        bounds = ub_lb
    )
    
    # remove the last coordinate, since that describes the radius of the ball
    return soln.x[:-1]

def find_interior_point(A,b,lb,ub):
    c = np.zeros(np.shape(A)[1])
    ub_lb = list(zip(lb,ub))
    soln = linprog(
        c=c,
        A=A,
        b_ub=b,
        bounds = ub_lb,
        method="highs-ipm"
    )
    return soln

def propose_rand_samples_hopsy(num_samples, init_point, path, lb, ub, dim, thin=10, threads=12):
    """
    sample a function using hopsy?
    """
    #TODO: add exception for if region is empty
    model = LAMCTS_hopsy_model(path, dim)
    global_constrs = path[0]
    constr_satisfied = True

    num_constrs = len(path)
    A_constr = np.zeros((num_constrs,dim))
    b_constr = np.zeros(num_constrs)
    global_constrs = path[0]
    # we will add the inequality constraint using the hopsy function
    if global_constrs["A_ineq"] is not None and global_constrs["b_ineq"] is not None:
        A = global_constrs["A_ineq"]
        b = global_constrs["b_ineq"]
    else: # no global constraint
        A = np.array([np.ones(dim)])
        b = np.array([np.inf])

    # let the global constraint occupy the first spot...
    A_constr[0,:] = A[0]
    b_constr[0] = b[0]

    # extract constraints from svc objects
    for i in range(1,len(path)):
        # TODO: want to be able to change based on kernel type?
        # path is a list of tuples (classifier,0)
        this_classifier = path[i][0].classifier.svm
        # matrix with single row, so extract the vector
        coefs = this_classifier.coef_[0]
        # vector with single entry, so extract the scalar
        intercept = this_classifier.intercept_[0]
        # the set of positive examples for the svm is
        # coefs^T x + intercept >= 0
        # so to rewrite this in hopsy form,
        # -coefs^T x <= intercept
        # oh I guess it was the other way around (the set of "good" examples is labeled by 0)
        A_constr[i,:] = coefs
        b_constr[i] = -intercept

    problem = hopsy.Problem(A_constr, b_constr, model)
    problem = hopsy.add_box_constraints(problem=problem,lower_bound=lb,upper_bound=ub)
    
    # if no point supplied: find the chebyshev center
    if init_point is None:
        init_point = find_interior_point(A_constr,b_constr,lb,ub)

    # try making one markov chain for each thread?
    # mc = hopsy.MarkovChain(problem, 
    #     proposal=hopsy.UniformCoordinateHitAndRunProposal, 
    #     starting_point=init_point)
    mc = [
        hopsy.MarkovChain(
            problem, proposal=hopsy.DikinWalkProposal, starting_point=init_point
        )
        for i in range(threads)
    ]
    # randomly generate the starting seed
    rng = [hopsy.RandomNumberGenerator(seed=np.random.randint(100)) for i in range(threads)]
    acceptance_rate, states = hopsy.sample(mc, rng, 
        n_samples=num_samples, 
        thinning=thin,
        n_threads=threads)
    print("hopsy sampler acceptance rate:",acceptance_rate)
    # nested 2d array? idk
    return acceptance_rate, states[0]
