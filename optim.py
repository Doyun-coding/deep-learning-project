import numpy as np, math

class SGD:
    def __init__(self, lr=0.01): self.lr=lr
    def update(self, params, grads):
        for k,g in grads.items(): params[k] -= self.lr * g

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self.m={}; self.v={}; self.t=0
    def update(self, params, grads):
        self.t += 1
        lr_t = self.lr * math.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for k,g in grads.items():
            if k not in self.m:
                self.m[k]=np.zeros_like(params[k]); self.v[k]=np.zeros_like(params[k])
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*(g**2)
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)
