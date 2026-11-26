import numpy as np

import torch.nn as nn





def unbroadcast(grad, shape):
    """ Shrink grad to `shape` after broadcasting in the forward pass."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, parents=(), requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self.parents = tuple(parents)
        self._backward = lambda: None

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    @staticmethod
    def ensure_tensor(value):
        if isinstance(value, Tensor):
            return value
        return Tensor(value, requires_grad=False)

    def __add__(self, other):
        other = Tensor.ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), 
        requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = backward
        return out

    def  backward(self):
        if self.data.size != 1:
            raise ValueError("backward() requires a scalar loss.")

        topo = []
        visited = set()
        def build_topology(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build_topology(parent)
                topo.append(node)

        build_topology(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __mul__(self, other):
        other = Tensor.ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), 
        requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = backward

        return out

    def __matmul__(self, other):
        other = Tensor.ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), 
        requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad @ other.data.T, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data.T @ out.grad, other.data.shape)
        out._backward = backward
        return out

    def __truediv__(self, other):
        other = Tensor.ensure_tensor(other)
        out = Tensor(self.data / other.data, (self, other), 
        requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            # out = a / b  →  ∂/∂a = 1/b ,  ∂/∂b = -a/b²
            if self.requires_grad:
                self.grad += unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(
                    -self.data / (other.data ** 2) * out.grad,
                    other.data.shape,
                )
        out._backward = backward
        return out

    def __pow__(self, other):
        other = Tensor.ensure_tensor(other)
        out = Tensor(self.data ** other.data, (self, other), 
        requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(
                    other.data * self.data ** (other.data - 1) * out.grad,
                    self.data.shape,
                )
            if other.requires_grad:
                other.grad += unbroadcast(
                    np.log(self.data) * out.data * out.grad,
                    other.data.shape,
                )
        out._backward = backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,),
        requires_grad=self.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)
        out._backward = backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data) / self.data.size
        out._backward = backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += unbroadcast((out.data > 0) * out.grad, self.data.shape)
        out._backward = backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += out.data * (1 - out.data) * out.grad
        out._backward = backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += (1 - out.data ** 2) * out.grad
        out._backward = backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += 1 / self.data * out.grad
        out._backward = backward
        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += 0.5 / np.sqrt(self.data) * out.grad
        out._backward = backward
        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += np.cos(self.data) * out.grad
        out._backward = backward
        return out

    def cos(self):
        out = Tensor(np.cos(self.data), (self,),
        requires_grad=self.requires_grad)
        def backward():
            if self.requires_grad:
                self.grad += -np.sin(self.data) * out.grad
        out._backward = backward
        return out

    def __radd__(self, other):
        return self.__add__(other)  

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return Tensor.ensure_tensor(other).__truediv__(self)

    def __rpow__(self, other):
        return Tensor.ensure_tensor(other)**self

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Tensor.ensure_tensor(other))

    def __rsub__(self, other):
        return Tensor.ensure_tensor(other) + (-self)


class MLP:
    """2-layer MLP: in → hidden (relu) → out. No nn.Module — just Tensors."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # small init so relu doesn't die immediately
        self.W1 = Tensor(rng.normal(scale=0.3, size=(n_in, n_hidden)).astype(np.float32), requires_grad=True)
        self.b1 = Tensor(np.zeros(n_hidden, dtype=np.float32), requires_grad=True)
        self.W2 = Tensor(rng.normal(scale=0.3, size=(n_hidden, n_out)).astype(np.float32), requires_grad=True)
        self.b2 = Tensor(np.zeros(n_out, dtype=np.float32), requires_grad=True)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        h = (X @ self.W1 + self.b1).relu()
        return h @ self.W2 + self.b2


def run_mlp(steps=1000, lr=0.05):
    """Fit y = 2*x0 - x1 + 0.5. Loss should fall if autograd + SGD work."""
    rng = np.random.default_rng(0)
    X_np = rng.normal(size=(64, 2)).astype(np.float32)
    y_np = (2.0 * X_np[:, 0] - X_np[:, 1] + 0.5).reshape(-1, 1).astype(np.float32)

    model = MLP(2, 8, 1)
    X = Tensor(X_np)
    y = Tensor(y_np)

    print("mlp smoke test (MSE should fall)")
    for step in range(steps):
        for p in model.parameters():
            p.zero_grad()

        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={float(loss.data):.6f}")

    return float(loss.data)


if __name__ == "__main__":
    # A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    # C = A @ B
    # # pretend L = sum(C), so G = ones like C
    # C.grad = np.ones_like(C.data)
    # C._backward()
    # print(A.grad)
    # # expected: ones(2,2) @ B.T = [[5+6, 7+8], [5+6, 7+8]] = [[11, 15], [11, 15]]
    # print(B.grad)
    # # expected: A.T @ ones = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]

    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    L = (A @ B).sum()
    L.backward()
    print(L.data)  # 19+22+43+50 = 134
    print(A.grad)  # [[11, 15], [11, 15]]
    print(B.grad)
    g = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    print(unbroadcast(g, ()))
    print(unbroadcast(g, (1,)))
    print(unbroadcast(g, (3,)))
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor(10.0, requires_grad=True)   # scalar, broadcasts
    L = (x + y).sum()
    L.backward()
    print(x.grad)  # [1, 1, 1]
    print(y.grad)  # 
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    L = (A @ B).mean()
    L.backward()
    print(L.data)  # 19+22+43+50 = 134
    print(A.grad)  # [[11, 15], [11, 15]]
    print(B.grad)  # [[11, 15], [11, 15]]
    print("--------------------------------")
    print('relu')
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.relu().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]
    print("--------------------------------")
    print('sigmoid')
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.sigmoid().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]
    print("--------------------------------")
    print("tanh")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.tanh().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]
    print("--------------------------------")
    print("exp")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.exp().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]
    print("--------------------------------")
    print("log")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.log().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print("--------------------------------")
    print("sqrt")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.sqrt().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]
    print("--------------------------------")
    print("sin")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.sin().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]
    print("--------------------------------")
    print("cos")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = A.cos().sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [1, 1, 1]  
    print("--------------------------------")
    print("division")
    a = Tensor([4.0, 9.0], requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    L = (a / b).sum()
    L.backward()
    print(a.grad, b.grad)  # [0.5 0.5]  -3.25
    x = Tensor(4.0, requires_grad=True)
    print((2 / x).data)   # 0.5  not 2.0
    print((2 ** x).data)  # 16.0 not 2.0
    print((2 + x).data)  # 6.0 not 2.0  
    print((2 * x).data)  # 8.0 not 2.0
    print("--------------------------------")
    print("neg")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = (-A).sum()
    L.backward()
    print(L.data)  # 1+2+3 = 6
    print(A.grad)  # [-1, -1, -1]
    print("--------------------------------")
    print("sub")
    A = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = (A - 2).sum()
    L.backward()
    print(L.data, A.grad)
    print("--------------------------------")
    run_mlp()

