import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import matplotlib.pyplot as plt

import unittest
import of.operators as op

class TestDeriveesImage(unittest.TestCase):
    def setUp(self):
        self.image_1 = np.arange(9).reshape(3, 3)


    def test_derivee_x(self):
        # A faire ecrire le corps de la fonction dans operators.py et le test
        pass

if __name__ == "__main__":
    unittest.main()
