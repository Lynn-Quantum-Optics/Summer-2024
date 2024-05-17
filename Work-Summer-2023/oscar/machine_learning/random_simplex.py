# heavily adapted from Severin Pappadeux from https://stackoverflow.com/questions/30658932/generating-n-uniform-random-numbers-that-sum-to-m
import numpy as np

def simplex_sampling(n):
    def make_corner_sample(n, k):
        r = []
        for i in range(0, n):
            if i == k:
                r.append(1.0)
            else:
                r.append(0.0)

        return r
    
    r = []
    for k in range(0,n):
        x = np.random.random()

        if x == 0.0:
            return make_corner_sample(n, k)
        
        t = -np.log(x)
        r.append(t)

    # normalize such that norm is 1
    r = np.array(r)
    r = r/np.linalg.norm(r)

    return r

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 8
    r = simplex_sampling(8)

    # make histrogram
    # do this many times
    samples = []
    for i in range(10000000):
        r = simplex_sampling(n)
        samples.append(r)

    samples = np.array(samples)
    plt.hist(samples[:,0], bins=1000)
    plt.savefig('simplex_hist.png')
    plt.show()

    

    