import numpy as np
from typing import Dict, List, Tuple
from layers import Affine, ReLU, SoftmaxWithLoss, Dropout, BatchNorm
from layers import Conv2D, MaxPool2x2, Flatten

def he_init(fan_in: int, fan_out: int, dtype=np.float32) -> np.ndarray:
    return (np.random.randn(fan_in, fan_out).astype(dtype) * np.sqrt(2.0 / max(1, fan_in)))

class MLP:
    def __init__(self, input_size:int, hidden_sizes:List[int], output_size:int,
                 use_batchnorm:bool=False, use_dropout:bool=False, dropout_ratio:float=0.5,
                 seed:int=42, weight_decay:float=0.0):
        np.random.seed(seed)
        self.use_bn=use_batchnorm; self.use_do=use_dropout
        sizes=[input_size]+list(hidden_sizes)+[output_size]
        self.params:Dict[str,np.ndarray]={}; self.layers=[]
        for i in range(len(sizes)-1):
            self.params[f"W{i+1}"]=he_init(sizes[i], sizes[i+1])
            self.params[f"b{i+1}"]=np.zeros(sizes[i+1], np.float32)
        for i in range(len(sizes)-2):
            self.layers.append(Affine(self.params[f"W{i+1}"], self.params[f"b{i+1}"]))
            if self.use_bn:
                self.params[f"gamma{i+1}"]=np.ones(sizes[i+1], np.float32)
                self.params[f"beta{i+1}"]=np.zeros(sizes[i+1], np.float32)
                bn=BatchNorm(sizes[i+1]); bn.gamma=self.params[f"gamma{i+1}"]; bn.beta=self.params[f"beta{i+1}"]
                self.layers.append(bn)
            self.layers.append(ReLU())
            if self.use_do: self.layers.append(Dropout(dropout_ratio))
        last=len(sizes)-1
        self.last_affine=Affine(self.params[f"W{last}"], self.params[f"b{last}"])
        self.last_layer=SoftmaxWithLoss()
        self.grads:Dict[str,np.ndarray] = {}

    def predict(self, x, train_flg=False):
        out=x
        for l in self.layers:
            out = l.forward(out, train_flg) if hasattr(l,'forward') and l.__class__.__name__ in ('BatchNorm','Dropout') else l.forward(out)
        return self.last_affine.forward(out)

    def loss(self, x, t, train_flg=True):
        return self.last_layer.forward(self.predict(x, train_flg), t)

    def backward(self):
        dout=self.last_layer.backward(1.0)
        dout=self.last_affine.backward(dout)
        w_total=len([k for k in self.params if k.startswith('W')])
        self.grads[f"W{w_total}"]=self.last_affine.dW; self.grads[f"b{w_total}"]=self.last_affine.db
        cur=w_total-1
        for l in self.layers[::-1]:
            dout=l.backward(dout)
            if isinstance(l, BatchNorm):
                self.grads[f"gamma{cur}"]=l.dgamma; self.grads[f"beta{cur}"]=l.dbeta
            if isinstance(l, Affine):
                self.grads[f"W{cur}"]=l.dW; self.grads[f"b{cur}"]=l.db; cur-=1

    def accuracy(self, X, y, batch_size=512):
        N=len(X); acc=0
        for i in range(0,N,batch_size):
            pred=np.argmax(self.predict(X[i:i+batch_size], False), axis=1)
            acc+=np.sum(pred==y[i:i+batch_size])
        return acc/N

    def params_and_grads(self): return self.params, self.grads

    def snapshot_state(self): return {k:v.copy() for k,v in self.params.items()}

    def load_state(self, state):
        for k,v in state.items(): self.params[k][...]=v

class CNN:
    def __init__(self, input_shape:Tuple[int,int,int], output_size:int=None,
                 hidden_fc:int=128, seed:int=42, use_dropout:bool=False, dropout_ratio:float=0.5, **kwargs):

        if output_size is None and "num_classes" in kwargs:
            output_size = kwargs.pop("num_classes")

        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword(s) for CNN: {unknown}")

        np.random.seed(seed)
        C,H,W = input_shape
        # conv1
        self.W1 = (np.random.randn(8, C, 3, 3).astype(np.float32) * np.sqrt(2.0/(C*3*3)))
        self.b1 = np.zeros(8, np.float32)
        self.conv1 = Conv2D(self.W1, self.b1, stride=1, pad=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2x2()
        self.W2 = (np.random.randn(16, 8, 3, 3).astype(np.float32) * np.sqrt(2.0/(8*3*3)))
        self.b2 = np.zeros(16, np.float32)
        self.conv2 = Conv2D(self.W2, self.b2, stride=1, pad=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2x2()
        self.flatten = Flatten()
        H2 = H//4; W2 = W//4; D_flat = 16*H2*W2
        self.W3 = he_init(D_flat, hidden_fc)
        self.b3 = np.zeros(hidden_fc, np.float32)
        self.fc1 = Affine(self.W3, self.b3)
        self.relu3 = ReLU()
        self.use_do = use_dropout
        self.drop = Dropout(dropout_ratio) if use_dropout else None
        self.W4 = he_init(hidden_fc, output_size)
        self.b4 = np.zeros(output_size, np.float32)
        self.fc2 = Affine(self.W4, self.b4)
        self.last = SoftmaxWithLoss()
        self.params = {"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,"W3":self.W3,"b3":self.b3,"W4":self.W4,"b4":self.b4}
        self.grads = {}

    def predict(self, x, train_flg=False):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        if self.use_do: out = self.drop.forward(out, train_flg)
        out = self.fc2.forward(out)
        return out

    def loss(self, x, t, train_flg=True):
        scores = self.predict(x, train_flg)
        return self.last.forward(scores, t)

    def backward(self):
        dout = self.last.backward(1.0)
        dout = self.fc2.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        _ = self.conv1.backward(dout)
        # collect grads
        self.grads = {
            "W1": self.conv1.dW, "b1": self.conv1.db,
            "W2": self.conv2.dW, "b2": self.conv2.db,
            "W3": self.fc1.dW,   "b3": self.fc1.db,
            "W4": self.fc2.dW,   "b4": self.fc2.db,
        }

    def accuracy(self, X, y, batch_size=128):
        N=len(X); acc=0
        for i in range(0,N,batch_size):
            pred = np.argmax(self.predict(X[i:i+batch_size], False), axis=1)
            acc += np.sum(pred == y[i:i+batch_size])
        return acc/N

    def params_and_grads(self): return self.params, self.grads

    def snapshot_state(self): return {k:v.copy() for k,v in self.params.items()}

    def load_state(self, state):
        for k,v in state.items(): self.params[k][...]=v
