import numpy as np
import copy

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    direction_dict = {
        "left": np.array([-1, 0]),
        "right": np.array([1, 0]),
        "up": np.array([0, 1]),
        "down": np.array([0, -1])}


    def histogram_filter(self, cmap, belief, action, observation): 
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.

        direction = {i for i in self.direction_dict if all(self.direction_dict[i] == action)}
        print(direction)
        # print("state T: ", self.state_transition_matrix(cmap,direction).shape)

        
        # intial alpha
        alpha = np.multiply(belief.reshape((-1)), self.sensor_matrix(cmap, observation))
        print("alpha shape: ", alpha.shape)
        print("T shape: ", self.state_transition_matrix(cmap, direction).shape)

        alpha = self.Forward(cmap, direction, observation, alpha) # update alpha

        eta = 1/(alpha.sum()) # calculate denominator

        belief = eta * alpha
        return belief.reshape(cmap.shape) #reshape back to size of cmap (20,20)

        
    def state_transition_matrix(self, cmap, direction):
        # direction: "left", "right", "up", "down"
        # return transformation matrix size (# states x # states) = (400x400)
        a = len(cmap)
        num_state = a**2
        T_matrix = np.identity(num_state)
        i, j = np.indices(T_matrix.shape)       
        if direction == {'left'}:
            T_matrix = T_matrix *0.1
            T_matrix[(i==j) & (i%a==0)] = 1
            T_matrix[i==j+1] = 0.9
            T_matrix[(i==j+1) & (i%a==0)] = 0
        elif direction == {'right'}:
            T_matrix = T_matrix *0.1
            T_matrix[(i==j) & (i%a==a-1)] = 1
            T_matrix[i==j-1] = 0.9
            T_matrix[(i==j-1) & (i%a==a-1)] = 0
        elif direction == {'up'}:
            T_matrix[(i==j) & (i>=a)] = 0.1
            T_matrix[i==j+a] = 0.9
        elif direction == {'down'}:
            T_matrix[(i==j) & (i < num_state-a)] = 0.1
            T_matrix[i==j-a] = 0.9           
        return T_matrix

    def sensor_matrix(self, cmap, observation):
        M_matrix = np.array(cmap, dtype ='float')
        # print(M)
        # print(observation)
        if observation == 1:
            M_matrix[cmap == 1] = 0.9
            M_matrix[cmap == 0] = 0.1
            # print("checking 1!")
        elif observation == 0:
            M_matrix[cmap == 0] = 0.9
            M_matrix[cmap == 1] = 0.1
            # print("checking 0!")
        # print("M: ", M)
        # return np.diag(M_matrix.reshape(-1)) # return size (# states x # states) = (400x400)
        return M_matrix.reshape(-1) # return M as a row vector (400,1)

    def Forward(self, cmap, direction, observation, alpha):
        """
        Update alpha in each iteration/timestep
        alpha: column vector of size (400,1)
        """
        
        alpha = np.multiply(self.sensor_matrix(cmap, observation), np.matmul(alpha, self.state_transition_matrix(cmap, direction)))

        return alpha



