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

        # Predict Step: calculate Predicted Belief from observations (no action, using previous belief)
        # predicted_bel = self.prob_dist_cmap(cmap, observation) * belief # element-wise multiplication
        direction = {i for i in self.direction_dict if all(self.direction_dict[i] == action)}
        print(direction)
        # print("state T: ", self.state_transition_matrix(cmap,direction).shape)
        predicted_belief = self.Predict(belief, direction)
        # print("predicted Belief: ", predicted_belief)

        # Update Step: calculate Belief from Predicted Belief and actions (using last observation)
        return self.Update(predicted_belief, cmap, observation)


        
    def state_transition_matrix(self, direction):
        # direction: "left", "right", "up", "down"
        # return transformation matrix size (# states x # states) = (400x400)
        T_matrix = np.identity(400)
        i, j = np.indices(T_matrix.shape)       
        if direction == {'left'}:
            T_matrix = T_matrix *0.1
            T_matrix[(i==j) & (i%20==0)] = 1
            T_matrix[i==j+1] = 0.9
            T_matrix[(i==j+1) & (i%20==0)] = 0
        elif direction == {'right'}:
            T_matrix = T_matrix *0.1
            T_matrix[(i==j) & (i%20==19)] = 1
            T_matrix[i==j-1] = 0.9
            T_matrix[(i==j-1) & (i%20==19)] = 0
        elif direction == {'up'}:
            T_matrix[(i==j) & (i>=20)] = 0.1
            T_matrix[i==j+20] = 0.9
        elif direction == {'down'}:
            T_matrix[(i==j) & (i < 400-20)] = 0.1
            T_matrix[i==j-20] = 0.9           
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
        return np.diag(M_matrix.reshape(-1)) # return size (# states x # states) = (400x400)

    def Predict(self, prior_belief, direction):
        # obtain direction from action in a step
        return np.matmul(self.state_transition_matrix(direction), prior_belief.reshape(-1))

    def Update(self, predicted_belief, cmap, observation):
        # print(predicted_belief.shape)
        # print(self.sensor_matrix(cmap, observation).shape)
        
        Belief = np.matmul(self.sensor_matrix(cmap, observation), predicted_belief)
        # print("Belief: ", Belief)
        return Belief.reshape((20,20)) / Belief.sum()




