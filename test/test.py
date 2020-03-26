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
        self.image_4 = np.array([4,5,7,1,0,6,3,3,2,8,4,1]).reshape(4,3)
        self.tab_1 = np.array([5,8,13,123,42,57,2,1,3]).reshape(3,3)
        self.d_tab_1 = cu.to_device(self.tab_1.astype(float))
        self.tab_2 = np.array([6,10,12,42,47,45,39,27,36]).reshape(3,3)
        self.matrice = np.array([[self.tab_1,self.tab_1],[self.tab_2,self.tab_2]])
        self.d_matrice = cu.to_device(self.matrice.astype(float))
        self.matrice2 = np.array([self.tab_1,self.tab_2])
        self.d_matrice2 = cu.to_device(self.matrice2.astype(float))
        self.image_5 = np.array([8,4,6,2,9,7,18,48,3]).reshape(3,3)
        self.d_image_1 = cu.to_device(self.image_1)
        self.d_image_2 = cu.to_device(self.image_2)
        self.d_image_4 = cu.to_device(self.image_4)
        self.d_image_5 = cu.to_device(self.image_5)
        self.d_tab_2 = cu.to_device(self.tab_2.astype(float))
        self.d_tab_dx = cu.device_array_like(self.d_image_1)
        self.d_tab_dy = cu.device_array_like(self.d_image_1)
        self.d_tab_dt = cu.device_array_like(self.d_image_1)
        self.d_somme_tab = cu.device_array_like(self.d_image_4)
        self.d_mul = cu.device_array_like(self.d_image_1)
        self.d_premier = cu.device_array_like(self.matrice[0,0].astype(float))
        self.d_deuxieme = cu.device_array_like(self.matrice[0,0].astype(float))
        self.d_determinant = cu.device_array_like(self.matrice[0,0].astype(float))
        self.d_matrice_inverse = cu.device_array_like(self.matrice.astype(float))

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
        return
        x, y = op.flux_optique(self.image_1, self.image_2, 1)[0]
    
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

    def test_multiplication_2_tab(self):
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.image_5.shape) + (BlockSize-1))//BlockSize
        mul_reference = self.tab_2*self.image_1
        op.multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](self.d_tab_2,self.d_image_1,1,self.d_mul)
        mul_GPU = self.d_mul.copy_to_host()
        self.assertTrue(np.all(mul_GPU == mul_reference))
    
    def test_inverser_matrice_GPU(self):
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.matrice[0,0].shape) + (BlockSize-1))//BlockSize
        matrice_reference = op.inverser_matrice(self.matrice)
        # print(self.matrice)
        # matrice_GPU = op.inverser_la_matrice_mi_GPU(self.matrice)
        op.inverser_la_matrice_GPU(self.d_matrice,self.d_premier,self.d_deuxieme,self.d_determinant, self.d_matrice_inverse)
        matrice_GPU = self.d_matrice_inverse.copy_to_host()
        self.assertTrue(np.all(matrice_GPU == matrice_reference))

    def test_somme_fenetre_global_GPU(self):
        # return 0
        rayon = 4
        BlockSize = np.array([32,32])
        gridSize = (np.asarray(self.image_4.shape) + (BlockSize-1))//BlockSize
        somme_reference = op.somme_fenetre_global(self.image_4, rayon)
        op.somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](self.d_image_4, rayon, self.d_somme_tab)
        somme_GPU = self.d_somme_tab.copy_to_host()
        self.assertTrue(np.all(somme_GPU == somme_reference))

    def test_flot_optique_GPU(self):
        x_reference, y_reference = op.flux_optique(self.tab_1,self.tab_2,5)
        x_GPU, y_GPU = op.flux_optique_GPU(self.d_tab_1,self.d_tab_2,5,self.d_matrice,self.matrice2)
        print(1)

if __name__ == "__main__":
    unittest.main()

