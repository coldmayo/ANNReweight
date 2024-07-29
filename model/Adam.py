try:
    import cupy as np
except ImportError:
    import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.0025, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param_update = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(param_update)

        return updated_params