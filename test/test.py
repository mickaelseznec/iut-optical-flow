import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import matplotlib.pyplot as plt

import unittest
import of.operators as op

class TestDeriveesImage(unittest.TestCase):
    def setUp(self):
        self.image_1 = np.array([5,3,1,6,8,0,1,2,3]).reshape(3, 3)
        self.image_2 = np.array([4,5,7,1,0,6,3,3,2]).reshape(3,3)
        self.image_3 = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,1,1,1,1,2,6,3,8,4]).reshape(5,5)

    def test_derivee_x(self):
        # A faire ecrire le corps de la fonction dans operators.py et le test
        derivee_x_reference = np.array([[-2,-2,0],[2,-8,0],[1,1,0]])
        derivee_x = op.derivee_x(self.image_1)
        self.assertTrue(np.all(derivee_x == derivee_x_reference))

    def test_derivee_y(self):

        # A faire ecrire le corps de la fonction dans operators.py et le test
        derivee_y_reference = np.array([[-3,-5,-1],[2,3,-4],[0,0,0]])
        derivee_y = op.derivee_y(self.image_2)
        self.assertTrue(np.all(derivee_y == derivee_y_reference))
   
    def test_derivee_t(self):
        # A faire ecrire le corps de la fonction dans operators.py et le test
        derivee_t_reference = np.array([[-1,2,6],[-5,-8,6],[2,1,-1]])
        derivee_t = op.derivee_t(self.image_1,self.image_2)
        self.assertTrue(np.all(derivee_t== derivee_t_reference))   

    def test_somme_fenetre(self):
        somme = op.somme_fenetre(self.image_3,0,0,3)
        # print(somme)
    
    def test_flot_optique(self):
        # print(self.image_1)
        op.flux_optique(self.image_1, self.image_2)
        # pass


if __name__ == "__main__":
    unittest.main()

