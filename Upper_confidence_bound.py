# Problem statement: Find out the best ad


# Importing the libraries

import math
import matplotlib.pyplot as plt
import pandas as pd


# Implementing UCB

class UpperConfidenceBound:

    def __init__(self, n, d, path):
        self.path = path
        self.N = n
        self.d = d
        self.dataset = pd.read_csv(path)
        self.ads_selected = []
        self.numbers_of_selections = [0] * self.d
        self.sums_of_rewards = [0] * self.d
        self.total_reward = 0

    def implement_ucb(self):

        for n in range(0, self.N):
            ad = 0
            max_upper_bound = 0
            for i in range(0, self.d):
                if self.numbers_of_selections[i] > 0:
                    average_reward = self.sums_of_rewards[i] / self.numbers_of_selections[i]
                    delta_i = math.sqrt(3 / 2 * math.log(n + 1) / self.numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if (upper_bound > max_upper_bound):
                    max_upper_bound = upper_bound
                    ad = i
            self.ads_selected.append(ad)
            self.numbers_of_selections[ad] = self.numbers_of_selections[ad] + 1
            reward = self.dataset.values[n, ad]
            self.sums_of_rewards[ad] += reward
            self.total_reward = self.total_reward + reward

    def visualise(self):
        plt.hist(self.ads_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()


N = 10000
d = 10
path = "datasets/Ads_CTR_Optimisation.csv"
ucb = UpperConfidenceBound(N, d, path)
ucb.implement_ucb()
ucb.visualise()  # Visualising the results

# Improving the above model

N1 = 700
ucb_best = UpperConfidenceBound(N1, d, path)
ucb_best.implement_ucb()
ucb_best.visualise()

# observation:
# After 700 trails.itself we can findout that ad5 is the best ad
