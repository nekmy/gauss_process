import numpy as np
import matplotlib.pylab as plt

def gaussian_kernel(x0, x1, sigma=1.0):
    k = np.exp(-np.linalg.norm(x0 - x1, axis=-1)**2 / sigma)
    return k

class GaussianProcess(object):

    def __init__(self, x, t, alpha=1.0, beta=1.0, sigma=1.0):
        self.beta = beta
        self.sigma = sigma
        n = len(x)
        self.x = np.array(x)
        x = self.x.reshape((n, 1, -1))
        x_T = self.x.reshape((1, n, -1))
        self.C = alpha**-1 * gaussian_kernel(x, x_T, sigma) + beta**-1 * np.eye(n)
        self.t = t

    def sample(self, x):
        n = len(x)
        x = np.array(x).reshape((n, 1, -1))
        k = gaussian_kernel(x, np.expand_dims(self.x, axis=0), self.sigma)
        mu = np.dot(np.dot(k, np.linalg.inv(self.C)), self.t)
        c = gaussian_kernel(x.reshape(-1), x.reshape(-1), self.sigma) + self.beta**-1
        sigma = c - np.sum(np.dot(k, np.linalg.inv(self.C)) * k, axis=1)
        return mu, sigma

def main():
    x_data = (2.0*np.random.rand(20) - 1.0) * 2 * np.pi
    t = np.sin(x_data)
    alpha = 1.0
    beta = 1.0
    sigma = 1.0
    gaussian_process = GaussianProcess(x_data.reshape(-1, 1), t, alpha, beta, sigma)
    x = np.linspace(-2*np.pi, 2*np.pi, 101)
    mu, sigma = gaussian_process.sample(x.reshape(-1, 1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, mu, color="green")
    ax.fill_between(x, mu-sigma, mu+sigma, color="green", alpha=0.2)
    ax.scatter(x_data, t, marker="*", color="red")
    plt.show()

if __name__ == "__main__":
    main()