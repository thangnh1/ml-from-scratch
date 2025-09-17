import math
from numbers import Number


class Vector:
    __slots__ = ('_elements',)

    def __init__(self, elements):
        self._elements = tuple(elements)

    @property
    def elements(self):
        return list(self._elements)

    @property
    def size(self):
        return len(self._elements)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self._elements)

    def __getitem__(self, index):
        return self._elements[index]

    def __repr__(self):
        return f"Vector({self._elements})"

    def _check_vector(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        if self.size != other.size:
            raise ValueError("Vector sizes don't match")

    def __add__(self, other):
        self._check_vector(other)
        return Vector([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        self._check_vector(other)
        return Vector([a - b for a, b in zip(self, other)])

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Invalid operand type")
        return Vector([a * other for a in self])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        if not isinstance(scalar, Number):
            raise TypeError("Can only divide by a scalar (Number)")
        if scalar == 0:
            raise ZeroDivisionError("Division by zero")
        return Vector(a / scalar for a in self)

    def dot(self, other):
        self._check_vector(other)
        return math.fsum(a * b for a, b in zip(self, other))

    def norm(self, p=2):
        if not isinstance(p, Number):
            raise TypeError("p must be a number ≥ 1 or ∞")
        if p < 1 and not math.isinf(p):
            raise ValueError("p-norm is undefined for p < 1")
        if p == 1:
            return sum(abs(x) for x in self) # chuan 1 Manhattan
        elif p == 2:
            return math.sqrt(sum(x**2 for x in self)) # chuan 2 Euclid - do dai vector
        elif math.isinf(p):
            return max(abs(x) for x in self) # chuan vo cung
        else:
            return sum(abs(x)**p for x in self)**(1/p) # cong thuc tong quat

    def normalize(self, p=2):
        v_norm = self.norm(p)
        if v_norm == 0:
            raise ValueError("Cannot normalize the zero vector")
        return Vector([x/v_norm for x in self])
