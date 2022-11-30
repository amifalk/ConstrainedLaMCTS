import hopsy
import numpy as np
class LAMCTS_hopsy_model:
    def __init__(self, path,dim):
        self.path = path
        self.dim = dim
    def compute_negative_log_likelihood(self,x):
        path = self.path
        global_constrs = path[0]
        constr_satisfied = True
        # we will add the inequality constraint using the hopsy function
        # if global_constrs["A_ineq"] is not None and global_constrs["b_ineq"] is not None:
        #    A = global_constrs["A_ineq"]
        #    b = global_constrs["b_ineq"]
        #    constr_satisfied = constr_satisfied and np.all(A@x <= b)
        # could add branch for A_eq, but not necessary
        for node in path[1:]: # skip global constraint
            boundary = node[0].classifier.svm
            x_reshape = x.reshape(1,-1)
            constr_satisfied = constr_satisfied and boundary.predict(x_reshape) == node[1]
            # node[1] store the direction to go
        if constr_satisfied:
            return -np.log(1)
        else:
            return np.inf

def propose_rand_samples_hopsy(num_samples, init_point, path, lb, ub, dim, thin=10, threads=12):
    """
    sample a function using hopsy?
    """
    #TODO: add exception for if region is empty
    model = LAMCTS_hopsy_model(path, dim)
    global_constrs = path[0]
    constr_satisfied = True
    # we will add the inequality constraint using the hopsy function
    if global_constrs["A_ineq"] is not None and global_constrs["b_ineq"] is not None:
        A = global_constrs["A_ineq"]
        b = global_constrs["b_ineq"]
    else: # no global constraint
        A = np.array([np.ones(dim)])
        b = np.array([np.inf])
    problem = hopsy.Problem(A, b, model)
    problem = hopsy.add_box_constraints(problem=problem,lower_bound=lb,upper_bound=ub)

    # try making one markov chain for each thread?
    # mc = hopsy.MarkovChain(problem, 
    #     proposal=hopsy.UniformCoordinateHitAndRunProposal, 
    #     starting_point=init_point)
    mc = [
        hopsy.MarkovChain(
            problem, proposal=hopsy.UniformCoordinateHitAndRunProposal, starting_point=init_point
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