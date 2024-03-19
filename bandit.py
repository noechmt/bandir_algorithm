from random import randint, random
from math import sqrt, log
import matplotlib.pyplot as plt



def pull(arm):
    arms = [0.4, 0.8]
    return arms[arm]

def plot_cumulative_reward(name, N, cumulative_reward, offset):
    plt.ylabel("Cumulative reward")
    plt.xlabel("Number of pulls")
    match name:
        case "random":
            plt.plot(range(offset, N+offset), cumulative_reward)
        case "eps_greedy":
            plt.plot(range(offset, N + offset), cumulative_reward)
        case "eps_greedy_dec":
            plt.plot(range(offset, N + offset), cumulative_reward)
        case "UCB":
            plt.plot(range(offset, N + offset), cumulative_reward)
            
            
def bandit(K,N,name):
    s = [0 for i in range(K)]
    n = [1 for i in range(K)]
    cumulative_reward = [0 for i in range(N-K+1)]
    p = 0
    for i in range(K):
        r = pull(i)
        s[i] = r
    cumulative_reward[p] = sum(s[m] for m in range(K))
    p += 1
    for t in range(K+1,N+1):
        match name:
            case "random":
                im = randint(0, K-1)
            case "eps_greedy":
                eps = 0.1
                if random() < eps:
                    im = randint(0, K-1)
                else:
                    max = 0
                    for i in range(K):
                        temp = s[i]/n[i]
                        if temp > max:
                            max = temp
                            im = i
            case "eps_greedy_dec":
                eps = 1/log(t**2)
                if random() < eps:
                    im = randint(0, K-1)
                else:
                    max = 0
                    for i in range(K):
                        temp = s[i]/n[i]
                        if temp > max:
                            max = temp
                            im = i
            case "UCB":
                max = 0
                for i in range(K):
                    temp = s[i]/n[i] + sqrt((2*log(t) / n[i]))
                    if temp > max:
                        max = temp
                        im = i
        r = pull(im)
        s[im] += r
        n[im] += 1
        #for the plot
        cumulative_reward[p] = cumulative_reward[p-1] + r
        p += 1
    plot_cumulative_reward(name, N-K+1 ,cumulative_reward, K)
    return sum(s[m] for m in range(K))

K = 2
N = 100000

print(int(bandit(K,N,"random")))
print(int(bandit(K,N,"eps_greedy")))
print(int(bandit(K,N,"eps_greedy_dec")))
print(int(bandit(K,N, "UCB")))
#plt.show()
