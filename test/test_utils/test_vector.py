import unittest
import math

from ml_from_scratch.utils.math_utils import *

class TestVectorOperations(unittest.TestCase):

    def setUp(self):
        self.v1 = Vector([1, 2, 3])
        self.v2 = Vector([4, 5, 6])

    # ---------- Core ops ----------
    def test_addition(self):
        out = self.v1 + self.v2
        self.assertIsInstance(out, Vector)
        self.assertEqual(out.elements, [5, 7, 9])
        # original vectors should remain unchanged
        self.assertEqual(self.v1.elements, [1, 2, 3])
        self.assertEqual(self.v2.elements, [4, 5, 6])

    def test_subtraction(self):
        out = self.v2 - self.v1
        self.assertEqual(out.elements, [3, 3, 3])
        self.assertEqual(self.v1.elements, [1, 2, 3])
        self.assertEqual(self.v2.elements, [4, 5, 6])

    def test_scalar_multiplication(self):
        out = self.v1 * 2
        self.assertEqual(out.elements, [2, 4, 6])
        # negative & float scalar
        self.assertEqual((self.v1 * -0.5).elements, [-0.5, -1.0, -1.5])

    def test_scalar_multiplication_invalid_scalar_raises(self):
        with self.assertRaises(TypeError):
            _ = self.v1 * "2"  # non-numeric

    # ---------- Dot product ----------
    def test_dot_product(self):
        self.assertEqual(self.v1.dot(self.v2), 32)  # 1*4 + 2*5 + 3*6

    # ---------- Norm ----------
    def test_norm_l2_default(self):
        self.assertAlmostEqual(self.v1.norm(), math.sqrt(14.0), places=12)

    def test_norm_l1(self):
        self.assertEqual(self.v1.norm(p=1), 6)

    def test_norm_l3(self):
        # ||[1,2,3]||_3 = (1^3 + 2^3 + 3^3)^(1/3) = (36)^(1/3)
        self.assertAlmostEqual(self.v1.norm(p=3), 36 ** (1/3), places=12)

    def test_norm_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            _ = self.v1.norm(p=0)  # p-norm undefined for p <= 0

    # ---------- Normalize ----------
    def test_normalize(self):
        u = self.v1.normalize()
        self.assertIsInstance(u, Vector)
        # unit length
        self.assertAlmostEqual(u.norm(), 1.0, places=12)
        # direction preserved
        n = math.sqrt(14.0)
        expected = [1/n, 2/n, 3/n]
        for a, b in zip(u.elements, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_normalize_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            _ = Vector([0, 0, 0]).normalize()

    # ---------- Shape & validation ----------
    def test_size_property(self):
        self.assertEqual(self.v1.size, 3)
        self.assertEqual(Vector([]).size, 0)

    def test_addition_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _ = self.v1 + Vector([1, 2])

    def test_subtraction_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _ = self.v1 - Vector([1, 2])

    def test_dot_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _ = self.v1.dot(Vector([1, 2]))

    # ---------- Sanity/behavior ----------
    def test_immutability_of_operands(self):
        _ = (self.v1 + self.v2)
        _ = (self.v1 - self.v2)
        _ = (self.v1 * 3)
        _ = self.v1.dot(self.v2)
        _ = self.v1.norm()
        # ensure original vectors not mutated
        self.assertEqual(self.v1.elements, [1, 2, 3])
        self.assertEqual(self.v2.elements, [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
