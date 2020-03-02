import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from numba import cuda
import numpy as np
import numba.cuda as cu
import matplotlib.pyplot as plt
import scipy.signal
import unittest
import of.operators as op

class TestDeriveesImage(unittest.TestCase):
    def setUp(self):
        self.image_1 = np.array([5,3,1,6,8,0,1,2,3]).reshape(3, 3)
        self.image_2 = np.array([4,5,7,1,0,6,3,3,2]).reshape(3,3)
        self.image_3 = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,1,1,1,1,2,6,3,8,4]).reshape(5,5)
        self.d_image_1 = cu.to_device(self.image_1)
        self.d_image_2 = cu.to_device(self.image_2)
        self.d_tab_dx = cu.device_array_like(self.d_image_1)
        self.d_tab_dy = cu.device_array_like(self.d_image_1)
        self.d_tab_dt = cu.device_array_like(self.d_image_1)
        self.d_somme_tab = cu.device_array_like(self.d_image_1)

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
        somme_reference = 156
        somme = op.somme_fenetre(self.image_1,0,0,3)
        self.assertTrue(np.all(somme==somme_reference))
    
    def test_somme_fenetre_global(self):
        somme_tab_reference = np.array([[91,75,59],[80,70,60],[69,65,61]])
        rayon = 2
        somme_tab = op.somme_fenetre_global(self.image_1,rayon)
        self.assertTrue(np.all(somme_tab == somme_tab_reference))   

    def test_flot_optique(self):
        # print(self.image_1)
        # non = op.flux_optique(self.image_1, self.image_2, 1)[0]
        # print(non)
        pass
    
    def test_derivee_x_GPU(self):
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.image_1.shape) + (BlockSize-1))//BlockSize
        derivee_x_reference = op.derivee_x(self.image_1)
        op.derivee_x_GPU[list(gridSize), list(BlockSize)](self.d_image_1,self.d_tab_dx)
        derivee_x_GPU = self.d_tab_dx.copy_to_host()
        self.assertTrue(np.all(derivee_x_GPU == derivee_x_reference))

    def test_derivee_y_GPU(self):
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.image_1.shape) + (BlockSize-1))//BlockSize
        derivee_y_reference = op.derivee_y(self.image_1)
        op.derivee_y_GPU[list(gridSize), list(BlockSize)](self.d_image_1,self.d_tab_dy)
        derivee_y_GPU = self.d_tab_dy.copy_to_host()
        self.assertTrue(np.all(derivee_y_GPU == derivee_y_reference))

    def test_derivee_t_GPU(self):
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.image_1.shape) + (BlockSize-1))//BlockSize
        derivee_t_reference = op.derivee_t(self.image_1, self.image_2)
        op.derivee_t_GPU[list(gridSize), list(BlockSize)](self.d_image_1, self.d_image_2, self.d_tab_dt)
        derivee_t_GPU = self.d_tab_dt.copy_to_host()
        self.assertTrue(np.all(derivee_t_GPU == derivee_t_reference))

    def test_somme_fenetre_global_GPU(self):
        rayon = 1
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.image_1.shape) + (BlockSize-1))//BlockSize
        somme_reference = op.somme_fenetre_global(self.image_1, rayon)
        op.somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](self.d_image_1, rayon, self.d_somme_tab)
        somme_GPU = self.d_somme_tab.copy_to_host()
        self.assertTrue(np.all(somme_GPU == somme_reference))

if __name__ == "__main__":
    unittest.main()

