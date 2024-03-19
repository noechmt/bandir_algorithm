import numpy as np
import math
import matplotlib.pyplot as plt



def pull(arm):
    arms = [0.6, 0.8]
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
    s = np.zeros(K)
    n = np.ones(K)
    cumulative_reward = np.zeros(N-K+1)
    p = 0
    for i in range(K):
        r = pull(i)
        s[i] = r
    cumulative_reward[p] = sum(s[m] for m in range(K))
    p += 1
    for t in range(K+1,N+1):
        match name:
            case "random":
                im = np.random.randint(K)
            case "eps_greedy":
                eps = 0.1
                if np.random.random() < eps:
                    im = np.random.randint(K)
                else:
                    im = np.argmax(s/n)
            case "eps_greedy_dec":
                eps = math.log(1/t**2)
                if np.random.random() < eps:
                    im = np.random.randint(K)
                else:
                    im = np.argmax(s/n)
            case "UCB":
                max = 0
                for i in range(K):
                    temp = s[i]/n[i] + math.sqrt((2*math.log(t) / n[i]))
                    if temp > max:
                        max = temp
                        indice = i
                im = indice
        r = pull(im)
        s[im] += r
        n[im] += 1
        #for the plot
        cumulative_reward[p] = cumulative_reward[p-1] + r
        p += 1
    plot_cumulative_reward(name, N-K+1 ,cumulative_reward, K)
    return sum(s[m] for m in range(K))
        
print(int(bandit(2,100000,"random")))
print(int(bandit(2,100000,"eps_greedy")))
print(int(bandit(2,100000,"eps_greedy_dec")))
print(int(bandit(2, 100000, "UCB")))
plt.show()
