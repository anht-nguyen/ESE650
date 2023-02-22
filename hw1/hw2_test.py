import numpy as np
import HMM_solution
import copy
if __name__ == "__main__":
    
    # Load the data
    M = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
    T = np.ones((2,2)) * 0.5
    pi_init = np.array([0.5, 0.5])

    obs_dict = {"LA": 0, "NY": 1, "null": 2}
    obs_keys = ["null", "LA", "LA", "null", "NY", "null", "NY", "NY", 
        "NY", "null", "NY", "NY", "NY", "NY", "NY", "null", "null", "LA", "LA", "NY"]
    observations = np.array([obs_dict[x] for x in obs_keys])
    # print(observations)

    HMM = HMM_solution.HMM(observations, copy.copy(T), copy.copy(M), pi_init)
    alpha = HMM.forward()

    beta = HMM.backward()
    gamma = HMM.gamma_comp(alpha, beta)
    xi = HMM.xi_comp(alpha, beta, gamma)
    print("alpha: ", alpha)
    print("beta: ", beta)
    print("gamma: ", gamma)
    print(np.sum(gamma, axis=0))
    print("\nxi: ", xi)
    print(np.sum(xi, axis=0))

    T_prime, M_prime, new_init_state = HMM.update(alpha, beta, gamma, xi)
    print("T prime: ", T_prime)
    print("M prime: ", M_prime)

    # traj_idx = [np.argmax(gamma[:,k]) for k in range(len(observations))]
    # traj = ["LA" if item == 0 else ("NY" if item == 1 else "null") for item in traj_idx]
    print(traj)
    print("DONE")
    # for i in range(len(observations)):


    # def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

    #     # P_original = 0
    #     # P_prime = 0

    #     return 0, 0
