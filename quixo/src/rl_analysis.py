from player import Player, RLPlayer
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

fr = open("../policies/policy_rl_player2_advance", 'rb')
dictionary = pickle.load(fr)
fr.close()

d = np.array(list(dictionary.values())).reshape((1, len(dictionary)))
mu = d.mean(1)
print(mu)
dc = d - mu
var = (1/dc.shape[0])*np.dot(dc, dc.T)
sigma = math.sqrt(var)

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.grid()
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.savefig("../policies/policy_player2_advance.png")


# 2k games ~ 3.8kb for player1
# 2k games ~ 3kb for player2
# 2.5k games doesn't change the size that much
