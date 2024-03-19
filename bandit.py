from random import randint, random
from math import sqrt, log
from time import time
import matplotlib.pyplot as plt

number_of_try = 1
K = 2
N = 100000
cumulative_reward = [0 for i in range(10*number_of_try)]
p = 0
list_display = [x*N/10 for x in range(1, 11)]

def pull(arm):
    arms = [0.4, 0.8]
    return arms[arm]

def mean_cumulative_reward(cumlative_reward):
    average = [0 for i in range(10)]
    for i in range(10):
        average[i] = sum(cumulative_reward[i+x] for x in range(0,number_of_try,10))/10
    return average

def plot_cumulative_reward(name, cumulative_reward, list_display):
    plt.figure("reward")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Number of pulls")
    match name:
        case "random":
            plt.plot(list_display, cumulative_reward, "-r^", label='random')
        case "eps_greedy":
            plt.plot(list_display, cumulative_reward, "-go", label='eps_greedy')
        case "eps_greedy_dec":
            plt.plot(list_display, cumulative_reward, "-yv", label='eps_greedy_dec')
        case "UCB":
            plt.plot(list_display, cumulative_reward, "-bs", label='UCB')
          
            
def plot_time(name, duree):
    plt.figure("time")
    plt.ylabel("Time (in seconds)")
    plt.xlabel("Number of pulls")
    match name:
        case "random":
            plt.plot(range(len(duree)), duree)
        case "eps_greedy":
            plt.plot(range(len(duree)), duree)
        case "eps_greedy_dec":
            plt.plot(range(len(duree)), duree)
        case "UCB":
            plt.plot(range(len(duree)), duree)

def bandit(K,N,name):
    s = [0 for i in range(K)]
    n = [1 for i in range(K)]
    global cumulative_reward
    global p
    for i in range(K):
        r = pull(i)
        s[i] = r
    #cumulative_reward[p] = sum(s[m] for m in range(K))
    #p += 1
    # variable de temps
    start = time()
    duree = []
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

        if(t%10000) == 0:
            duree.append(time()-start)

        r = pull(im)
        s[im] += r
        n[im] += 1
        if(t in list_display):
            cumulative_reward[p] = sum(s[m] for m in range(K)) 
            p += 1      
        #for the plot
        #cumulative_reward[p] = cumulative_reward[p-1] + r
        #p += 1

    
    plot_time(name, duree)
    return sum(s[m] for m in range(K))



#print(int(bandit(K,N,"random")))
for i in range(number_of_try):
    bandit(K,N,"random")
plot_cumulative_reward("random",mean_cumulative_reward(cumulative_reward), list_display)
p = 0
for i in range(number_of_try):
    bandit(K,N,"eps_greedy")
plot_cumulative_reward("eps_greedy",mean_cumulative_reward(cumulative_reward), list_display)
p = 0
for i in range(number_of_try):
    bandit(K,N,"eps_greedy_dec")
plot_cumulative_reward("eps_greedy_dec",mean_cumulative_reward(cumulative_reward), list_display)
p = 0
for i in range(number_of_try):
    bandit(K,N,"UCB")
plot_cumulative_reward("UCB",mean_cumulative_reward(cumulative_reward), list_display)
plt.legend()
plt.show()
