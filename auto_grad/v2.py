import numpy as np


def unbroadcast(grad, shape):
    """Reduce a broadcasted gradient back to its original shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, parents=(), op="", requires_grad=False):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self._prev = set(parents)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    
    @staticmethod
    def ensure_tensor(value):
        if isinstance(value, Tensor):
            return value
        return Tensor(value, requires_grad=False)

    def __add__(self, other):
        other = Tensor.ensure_tensor(other)
        needs_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), "+", needs_grad)

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Tensor.ensure_tensor(other))

    def __rsub__(self, other):
        return Tensor.ensure_tensor(other) + (-self)

    def __mul__(self, other):
        other = Tensor.ensure_tensor(other)
        needs_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), "*", needs_grad)

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = Tensor.ensure_tensor(other)
        needs_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other), "@", needs_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), "transpose", self.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad.T
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu", self.requires_grad)
        
        def backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), "sigmoid", self.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad * (1 - out.data) * out.data
        out._backward = backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log", self.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), "mean", self.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data) / self.data.size
        out._backward = backward
        return out

    def backward(self):
        if self.data.size != 1:
            raise ValueError("backward() requires a scalar loss.")

        topo = []
        visited = set()
        def build_topology(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topology(parent)
                topo.append(node)

        build_topology(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
