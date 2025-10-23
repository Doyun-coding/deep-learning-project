import numpy as np

def im2col(x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    out_h = (H + 2*pad - FH)//stride + 1
    out_w = (W + 2*pad - FW)//stride + 1
    img = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N, C, FH, FW, out_h, out_w), dtype=x.dtype)
    for y in range(FH):
        y_max = y + stride*out_h
        for x_ in range(FW):
            x_max = x_ + stride*out_w
            cols[:, :, y, x_, :, :] = img[:, :, y:y_max:stride, x_:x_max:stride]
    cols = cols.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return cols, out_h, out_w

def col2im(cols, x_shape, FH, FW, stride=1, pad=0, out_h=None, out_w=None):
    N, C, H, W = x_shape
    if out_h is None or out_w is None:
        out_h = (H + 2*pad - FH)//stride + 1
        out_w = (W + 2*pad - FW)//stride + 1
    cols = cols.reshape(N, out_h, out_w, C, FH, FW).transpose(0,3,4,5,1,2)
    img = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=cols.dtype)
    for y in range(FH):
        y_max = y + stride*out_h
        for x_ in range(FW):
            x_max = x_ + stride*out_w
            img[:, :, y:y_max:stride, x_:x_max:stride] += cols[:, :, y, x_, :, :]
    return img[:, :, pad:H+pad, pad:W+pad]

class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.x = None; self.dW = None; self.db = None
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

class ReLU:
    def __init__(self): self.mask=None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy(); out[self.mask]=0
        return out
    def backward(self, dout):
        dout[self.mask]=0
        return dout

class Dropout:
    def __init__(self, p=0.5):
        self.p = p; self.mask=None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
            return x*self.mask/(1.0-self.p)
        else:
            return x
    def backward(self, dout):
        return dout*self.mask/(1.0-self.p)

class BatchNorm:
    def __init__(self, D, momentum=0.9, eps=1e-5):
        self.gamma=np.ones(D, np.float32); self.beta=np.zeros(D, np.float32)
        self.momentum=momentum; self.eps=eps
        self.running_mean=np.zeros(D, np.float32); self.running_var=np.ones(D, np.float32)
        self.xc=None; self.std_inv=None; self.x_hat=None
        self.dgamma=None; self.dbeta=None
    def forward(self, x, train_flg=True):
        if train_flg:
            mu=x.mean(axis=0); var=x.var(axis=0)
            self.xc=x-mu; self.std_inv=1.0/np.sqrt(var+self.eps); self.x_hat=self.xc*self.std_inv
            out=self.gamma*self.x_hat+self.beta
            self.running_mean=self.momentum*self.running_mean+(1-self.momentum)*mu
            self.running_var =self.momentum*self.running_var +(1-self.momentum)*var
            return out
        else:
            x_hat=(x-self.running_mean)/np.sqrt(self.running_var+self.eps)
            return self.gamma*x_hat+self.beta
    def backward(self, dout):
        N,_=dout.shape
        self.dbeta=dout.sum(axis=0)
        self.dgamma=np.sum(dout*self.x_hat, axis=0)
        dxhat=dout*self.gamma
        dvar=np.sum(dxhat*self.xc*(-0.5)*(self.std_inv**3), axis=0)
        dmu =np.sum(-dxhat*self.std_inv, axis=0)+dvar*np.mean(-2.0*self.xc, axis=0)
        dx  =dxhat*self.std_inv + dvar*2.0*self.xc/N + dmu/N
        return dx

class SoftmaxWithLoss:
    def __init__(self): self.y=None; self.t=None
    def forward(self, scores, t):
        z=scores - np.max(scores,axis=1,keepdims=True)
        expz=np.exp(z); self.y=expz/np.sum(expz,axis=1,keepdims=True)
        self.t=t.astype(np.int64)
        N=scores.shape[0]
        return -np.log(self.y[np.arange(N), self.t] + 1e-12).mean()
    def backward(self, dout=1.0):
        N=self.y.shape[0]
        dx=self.y.copy(); dx[np.arange(N), self.t]-=1.0; dx/=N
        return dx*dout


class Conv2D:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b = W, b
        self.stride, self.pad = stride, pad
        self.x_shape=None; self.cols=None
        self.dW=None; self.db=None

    def forward(self, x):
        self.x_shape = x.shape
        F, C, FH, FW = self.W.shape
        cols, out_h, out_w = im2col(x, FH, FW, self.stride, self.pad)
        self.cols = cols
        Wcol = self.W.reshape(F, -1).T
        out = self.cols @ Wcol + self.b
        out = out.reshape(self.x_shape[0], out_h, out_w, F).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        N, F, Oh, Ow = dout.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, F)
        self.db = dout.sum(axis=0)
        Wcol = self.W.reshape(F, -1).T
        dcols = dout @ Wcol.T
        self.dW = (self.cols.T @ dout).T.reshape(self.W.shape)
        dx = col2im(dcols, self.x_shape, self.W.shape[2], self.W.shape[3],
                    self.stride, self.pad, out_h=Oh, out_w=Ow)
        return dx

class MaxPool2x2:
    def __init__(self):
        self.x_shape=None; self.argmax=None; self.cols_shape=None

    def forward(self, x):
        N, C, H, W = x.shape
        FH=FW=2; stride=2; pad=0
        x_ = x.reshape(N*C, 1, H, W)
        cols, out_h, out_w = im2col(x_, FH, FW, stride, pad)
        cols = cols.reshape(-1, FH*FW)
        self.argmax = np.argmax(cols, axis=1)
        out = cols.max(axis=1).reshape(N, C, out_h, out_w)
        self.x_shape = x.shape
        self.cols_shape = (N, C, out_h, out_w)
        return out

    def backward(self, dout):
        N, C, Oh, Ow = dout.shape
        FH=FW=2; stride=2; pad=0
        dcols = np.zeros((N*C*Oh*Ow, FH*FW), dtype=dout.dtype)
        dcols[np.arange(dcols.shape[0]), self.argmax] = dout.reshape(-1)
        dcols = dcols.reshape(N*C*Oh*Ow, -1)
        dX = col2im(dcols, (N*C,1,self.x_shape[2],self.x_shape[3]), FH, FW, stride, pad, out_h=Oh, out_w=Ow)
        return dX.reshape(self.x_shape)

class Flatten:
    def __init__(self): self.in_shape=None

    def forward(self, x):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.in_shape)
