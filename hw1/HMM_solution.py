import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations # size (20,)
        self.Transition = Transition # size (2,2)
        self.Emission = Emission # M: observation matrix of size (2,3)
        self.Initial_distribution = Initial_distribution # size (2,)
        print("check init")

    def forward(self):
        print("check forward")
        alpha = np.array([np.multiply(self.Initial_distribution, self.Emission[:,self.Observations[0]])] ).reshape(len(self.Initial_distribution), 1)# element-wise multiplication

        for i in range(len(self.Observations)-1):
            alpha = np.append(alpha, 
                np.multiply(self.Emission[:,self.Observations[i+1]], np.matmul(alpha[:,i], self.Transition) ).reshape(len(self.Initial_distribution),1) , 
                axis = 1)
        return alpha.transpose()

    def backward(self):
        print("check backward")
        beta = np.ones((len(self.Initial_distribution),1))
        for i in range(len(self.Observations)-1): #count the number of observations (20) reversely
            beta = np.append( np.matmul( np.multiply(beta[:,-i-1], self.Emission[:, self.Observations[-i-1]]), self.Transition).reshape((len(self.Initial_distribution), 1)),
                beta,
                axis = 1)
        return beta.transpose()

    def gamma_comp(self, alpha, beta): 
        alpha = alpha.transpose()
        beta = beta.transpose()
        print("check gamma")  
        gamma = np.hstack([np.multiply(alpha[:,k], beta[:,k]).reshape((len(self.Initial_distribution), 1))/ sum(alpha[:,-1])  for k in range(len(self.Observations)) ])
        return gamma.transpose()

    def xi_comp(self, alpha, beta, gamma):
        alpha = alpha.transpose()
        beta = beta.transpose()
        xi = np.hstack([ np.array(np.append([alpha[:,k]],[alpha[:,k]],axis=0).transpose().reshape(1,-1) * self.Transition.reshape(1,-1) * np.append(self.Emission[:,self.Observations[k+1]], self.Emission[:,self.Observations[k+1]]) * np.append(beta[:,k+1], beta[:,k+1])/ sum(alpha[:,-1])).reshape((len(self.Initial_distribution)*2, 1)) for k in range(len(self.Observations)-1)])
            # reshape [x,x'] as follow: [[0,0], [0,1], [1,0], [1,1]] (size (4,1))
        return xi.transpose().reshape((19,2,2))

    def update(self, alpha, beta, gamma, xi):
        gamma=gamma.transpose()
        xi = xi.reshape((19,4)).transpose()
        new_init_state = gamma[:,0]
        denom = 1/np.sum(gamma[:,:-1],axis=1)
        T_prime = np.sum(xi, axis=1) * np.append([denom],[denom],axis=0).transpose().reshape(1,-1)
        T_prime = T_prime.reshape((2,2))
        
        denom_M = 1/np.sum(gamma,axis=1)
        print(denom_M)
        numer_M = np.vstack([np.sum(gamma[:, (self.Observations==y)], axis=1) for y in np.unique(self.Observations)]).transpose()
        print(numer_M)
        M_prime =  numer_M * np.stack((denom_M, denom_M, denom_M),axis=1)

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):
        P_original = ...
        P_prime = ...
        return P_original, P_prime
