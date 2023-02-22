import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random

if __name__ == "__main__":

    # Load the data
    data = np.load(open('starter.npz', 'rb'))
    cmap = data['arr_0']
    # plt.imshow(cmap)
    # plt.show()
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    # print("belief_states: \n", belief_states)
    print(belief_states.shape)


    #### Test your code here
    belief = 1/400 * np.ones((400,1)) #uniform initial belief
    # print(prior_belief)
    # print(prior_belief.sum())
    # print(cmap.sum())

    for i in range(30):
        print("i = ", i)
        # print("action: ", actions[i])
        # print("obs: ", observations[i])
        belief = HistogramFilter().histogram_filter(cmap, belief, actions[i], observations[i])
        print(belief.shape)

    # print(prior_belief)
    # print(prior_belief.sum())
    plt.imshow(belief)
    plt.show()