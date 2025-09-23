import math
from numbers import Number
import copy


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
        return Vector([a / scalar for a in self])

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


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows > 0 else 0

    @property
    def shape(self):
        return (self.rows, self.cols)

    def __repr__(self):
        return f"Matrix({self.data})"

    def __str__(self):
        if not self.data:
            return "Matrix([])"
        rows = ["[" + ", ".join(map(str, row)) + "]" for row in self.data]
        return "Matrix([\n  " + ",\n  ".join(rows) + "\n])"

    def _check_same_shape(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Operand must be a Matrix")
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape")

    def __add__(self, other):
        self._check_same_shape(other)
        return Matrix([a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data))


    def __sub__(self, other):
        self._check_same_shape(other)
        return Matrix([a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data))

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Can only multiply by a scalar (Number)")
        return Matrix([[other * x for x in row] for row in self.data])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Operand must be a Matrix")
        if self.cols != other.rows:
            raise ValueError("For A @ B, A.cols must equal B.rows")
        m, n, p = self.rows, self.cols, other.cols
        other_cols = list(zip(*other.data))
        result = [[sum(self.data[i][k] * other_cols[j][k] for k in range(n)) for j in range(p)] for i in range(m)]
        return Matrix(result)

    def transpose(self):
        return Matrix([list(col) for col in zip(*self.data)]) if self.rows and self.cols else Matrix([])


    def determinant(self): # detA
        if self.rows != self.cols:
            raise ValueError("Cannot compute determinant of non-square matrix")
        n = self.rows
        if n == 0:
            return 1 # det cua matrix rong

        A = copy.deepcopy(self.data)
        det_sign = 1
        eps = 1e-12 # hang so epsilon

        for k in range(n):
            pivot_row = max(range(k, n), key = lambda i: abs(A[i][k]))
            if abs(A[pivot_row][k]) < eps:
                return 0

            if pivot_row != k:
                det_sign *= -1
                A[pivot_row], A[k] = A[k], A[pivot_row]

            for i in range(k + 1, n):
                factor = A[i][k] / A[k][k]
                for j in range(k + 1, n):
                    A[i][j] -= factor * A[k][j]
                A[i][k] = 0
        det = det_sign
        for i in range(n):
            det *= A[i][i]
        return det

    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("Cannot compute inverse of non-square matrix")
        n = self.rows
        if n == 0:
            return Matrix([])

        aug = [self.data[i][:] + [0] * n for i in range(n)]
        for i in range(n):
            aug[i][n+i] = 1

        eps = 1e-12

        for k in range(n):
            pivot_row = max(range(k, n), key = lambda i: abs(aug[i][k]))
            if abs(aug[pivot_row][k]) < eps:
                raise ValueError("Matrix is singular and cannot be inverted")
            if pivot_row != k:
                aug[pivot_row], aug[k] = aug[k], aug[pivot_row]

            pivot_val = aug[k][k]
            inv_pivot = 1.0 / pivot_val
            for j in range(2 * n):
                aug[k][j] *= inv_pivot
            for i in range(n):
                if i == k:
                    continue
                factor = aug[i][k]
                if factor != 0.0:
                    for j in range(2 * n):
                        aug[i][j] -= factor * aug[k][j]
        inv_data = [row[n:] for row in aug]
        return Matrix(inv_data)

    def eigenvalue_power_iteration(self, max_iter=1000, tolerance=1e-8):
        """
        Find dominant eigenvalue and eigenvector using power iteration.

        Algorithm:
        1. Start with random vector v₀
        2. For k = 1, 2, ...:
           - v_k = A * v_{k-1}
           - v_k = v_k / ||v_k||  (normalize)
           - λ_k = v_k^T * A * v_k  (Rayleigh quotient)
        3. Stop when |λ_k - λ_{k-1}| < tolerance

        Returns: (eigenvalue, eigenvector, iterations)
        """
        if self.rows != self.cols:
            raise ValueError("Eigenvalues only for square matrices")

        n = self.rows

        # Initialize random vector
        import random
        random.seed(42)  # For reproducibility
        v = [random.random() - 0.5 for _ in range(n)]

        # Normalize initial vector
        norm = sum(x ** 2 for x in v) ** 0.5
        v = [x / norm for x in v]

        eigenvalue = 0

        for iteration in range(max_iter):
            # Matrix-vector multiplication: Av
            Av = [0] * n
            for i in range(n):
                for j in range(n):
                    Av[i] += self.data[i][j] * v[j]

            # Normalize Av
            norm = sum(x ** 2 for x in Av) ** 0.5
            if norm < tolerance:
                raise ValueError("Vector became zero - matrix might be singular")

            v_new = [x / norm for x in Av]

            # Calculate eigenvalue using Rayleigh quotient: v^T * A * v
            new_eigenvalue = 0
            for i in range(n):
                Av_i = sum(self.data[i][j] * v_new[j] for j in range(n))
                new_eigenvalue += v_new[i] * Av_i

            # Check convergence
            if iteration > 0 and abs(new_eigenvalue - eigenvalue) < tolerance:
                return new_eigenvalue, v_new, iteration + 1

            eigenvalue = new_eigenvalue
            v = v_new

        return eigenvalue, v, max_iter

    def all_eigenvalues_qr(self, max_iter=1000, tolerance=1e-8):
        """
        Find all eigenvalues using QR algorithm.

        Algorithm:
        1. A₀ = A
        2. For k = 1, 2, ...:
           - Q_k, R_k = QR(A_{k-1})
           - A_k = R_k * Q_k
        3. A_k converges to upper triangular (eigenvalues on diagonal)
        """
        if self.rows != self.cols:
            raise ValueError("QR algorithm only for square matrices")

        A = Matrix([row[:] for row in self.data])  # Deep copy
        n = self.rows

        for iteration in range(max_iter):
            # QR decomposition
            Q, R = A.qr_decomposition()

            # A = R * Q
            A_new = R * Q

            # Check convergence (off-diagonal elements should be small)
            max_off_diagonal = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        max_off_diagonal = max(max_off_diagonal, abs(A_new.data[i][j]))

            if max_off_diagonal < tolerance:
                # Extract eigenvalues from diagonal
                eigenvalues = [A_new.data[i][i] for i in range(n)]
                return eigenvalues, iteration + 1

            A = A_new

        # Extract eigenvalues even if not fully converged
        eigenvalues = [A.data[i][i] for i in range(n)]
        return eigenvalues, max_iter

# if __name__ == "__main__":
#     a = Matrix([[2,3,4],[2,7,1],[0,1,4]])
#     b = Matrix([[1,2,3],[4,5,6],[7,8,9]])
#     print(a+b)