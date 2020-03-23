import numpy as np
import matplotlib.pyplot as plt
import numba.cuda as cu
import numpy as np
from PIL import Image


@cu.jit()
def derivee_x(d_image,d_tab_dx):

    # Retourne la derivée en x pour tous les pixels de l'image
    hauteur, largeur = d_image.shape
    h,l = cu.grid(2)
    if h >= hauteur or l >= largeur:
        return
    if l == largeur-1 :
        d_tab_dx[h,l] = 0
    else:
        d_tab_dx[h,l] = d_image[h,l+1] - d_image[h,l]
    
    

@cu.jit()
def derivee_y(d_image,d_tab_dy):
    # Retourne la derivée en y pour tous les pixels de l'image
    hauteur, largeur = d_image.shape
    h,l = cu.grid(2)
    if h >= hauteur or l >= largeur:
        return
    if h == hauteur-1 :
        d_tab_dy[h,l] = 0
    else:
        d_tab_dy[h,l] = d_image[h+1,l] - d_image[h,l]

    

@cu.jit()
def derivee_t(d_image_1, d_image_2,d_tab_dt):
    # Retourne la derivée en t pour tous les pixels de l'image
    h,l = cu.grid(2)
    d_tab_dt[h,l]  =  d_image_2[h,l] - d_image_1[h,l]
    



@cu.jit()
def somme_fenetre_global(d_image,d_r,d_som_tab):
    hauteur,largeur = d_image.shape
    h,l = cu.grid(2)
    result = 0
    if (h >= hauteur | l >= largeur):
        pass;
    else:
        for i in range (2*d_r+1):
            for j in range (2*d_r+1):
                indicel = l-d_r+j
                indiceh = h-d_r+i
                if(indiceh<0):
                    indiceh = 0
                elif(indiceh>=hauteur):
                    indiceh = hauteur-1
                if(indicel>=largeur):
                    indicel = largeur-1            
                elif(indicel<0):
                    indicel = 0
                result += d_image[indiceh,indicel]    
        d_som_tab[h,l] = result

@cu.jit()
def multiplication_2_tab(d_image1,d_image2,d_coef,d_mulp_tab):
    h,l = cu.grid(2)
    hauteur,largeur = d_image1.shape
    if h >= hauteur or l >= largeur:
        return
    d_mulp_tab[h,l] = d_image1[h,l] * d_coef*d_image2[h,l]
    
@cu.jit()
def division_2_tab(d_numerateur,d_denominateur,d_coef,d_tab_sortie):
    h,l = cu.grid(2)
    hauteur,largeur = d_image1.shape
    if h >= hauteur or l >= largeur:
        return
    d_tab_sortie[h,l] = d_coef*d_numerateur[h,l]/d_denominateur[h,l]

@cu.jit()
def soustraction_2_tab(d_image1,d_image2,d_tab_sortie):
    h,l = cu.grid(2)
    hauteur,largeur = d_image1.shape
    if h >= hauteur or l >= largeur:
        return
    d_tab_sortie[h,l] = d_image1[h,l] - d_image2[h,l]

@cu.jit()
def addition_2_tab(d_image1,d_image2,d_tab_sortie):
    h,l = cu.grid(2)
    hauteur,largeur = d_image1.shape
    if h >= hauteur or l >= largeur:
        return
    d_tab_sortie[h,l] = d_image1[h,l] + d_image2[h,l]
    
def inverser_la_matrice(d_matrice,d_premier,d_deuxieme,d_matrice_inverse):
    
    blockSize = np.array([32, 32])
   
    gridSize = (np.asarray(d_matrice.shape) + (blockSize - 1)) // blockSize
    
    multiplication_2_tab[tuple(gridSize), tuple(blockSize)](d_matrice[0,0],d_matrice[1,1],1.,d_premier)
    multiplication_2_tab[tuple(gridSize), tuple(blockSize)](d_matrice[0,1],d_matrice[1,0],1.,d_deuxieme)
    
    soustraction_2_tab[tuple(gridSize), tuple(blockSize)](d_premier, d_deuxieme, d_determinant)
    
    division_2_tab[tuple(gridSize), tuple(blockSize)](d_matrice[1,1],d_determinant,1.,d_matrice_inverse[0,0])
    division_2_tab[tuple(gridSize), tuple(blockSize)](d_matrice[0,1],d_determinant,-1.,d_matrice_inverse[0,1])
    division_2_tab[tuple(gridSize), tuple(blockSize)](d_matrice[1,0],d_determinant,-1.,d_matrice_inverse[1,0])
    division_2_tab[tuple(gridSize), tuple(blockSize)](d_matrice[0,0],d_determinant,1.,d_matrice_inverse[1,1])
    
    
    

def flot_optique(d_image1,d_image2,rayon):
    blockSize = np.array([50, 15])
    gridSize = (np.asarray(d_image1.shape) + (blockSize - 1)) // blockSize
    
    #création des tableau
    d_somme_tab_dx_2 = cu.device_array_like(d_image1)
    d_somme_tab_dy_2 = cu.device_array_like(d_image1)
    d_somme_tab_dx_dy = cu.device_array_like(d_image1)
    d_somme_tab_dx_dt = cu.device_array_like(d_image1)
    d_somme_tab_dy_dt = cu.device_array_like(d_image1)
    
   

    d_tab_dx = cu.device_array_like(d_image1)
    d_tab_dy = cu.device_array_like(d_image1)
    d_tab_dt = cu.device_array_like(d_image1)
    d_tab_dx_2 = cu.device_array_like(d_image1)
    d_tab_dy_2 = cu.device_array_like(d_image1)
    d_tab_dx_dy = cu.device_array_like(d_image1)
    d_tab_dy_dt = cu.device_array_like(d_image1)
    d_tab_dx_dt = cu.device_array_like(d_image1)
    d_premier = cu.device_array_like(d_image1)
    d_deuxieme = cu.device_array_like(d_image1)
    d_inv_tab = cu.device_array_like(d_image1)
    d_dx = cu.device_array_like(d_image1)
    d_dx1 = cu.device_array_like(d_image1)
    d_dx2 = cu.device_array_like(d_image1)
    d_dy = cu.device_array_like(d_image1)
    d_dy1 = cu.device_array_like(d_image1)
    d_dy2 = cu.device_array_like(d_image1)
    #calculer somme_tab_dx_2
    derivee_x[tuple(gridSize), tuple(blockSize)](d_image1,d_tab_dx)
    multiplication_2_tab[tuple(gridSize), tuple(blockSize)](d_tab_dx,d_tab_dx,1,d_tab_dx_2)
    somme_fenetre_global[tuple(gridSize), tuple(blockSize)](d_tab_dx_2,rayon,d_somme_tab_dx_2)
    
    #calculer somme_tab_dy_2
    derivee_y[tuple(gridSize), tuple(blockSize)](d_image1,d_tab_dy)
    multiplication_2_tab[tuple(gridSize), tuple(blockSize)](d_tab_dy,d_tab_dy,1,d_tab_dy_2)
    somme_fenetre_global[tuple(gridSize), tuple(blockSize)](d_tab_dy_2,rayon,d_somme_tab_dy_2)
    

    #calculer somme_tab_dx_dy

    multiplication_2_tab[tuple(gridSize), tuple(blockSize)](d_tab_dx,d_tab_dy,1,d_tab_dx_dy)
    somme_fenetre_global[tuple(gridSize), tuple(blockSize)](d_tab_dx_dy,rayon,d_somme_tab_dx_dy)
   

    #calculer somme_tab_dx_dt
    derivee_t[tuple(gridSize), tuple(blockSize)](d_image1,d_image2,d_tab_dt)
    multiplication_2_tab[tuple(gridSize), tuple(blockSize)](d_tab_dx,d_tab_dt,-1,d_tab_dx_dt)
    somme_fenetre_global[tuple(gridSize), tuple(blockSize)](d_tab_dx_dt,rayon,d_somme_tab_dx_dt)
    

    #calculer somme_tab_dy_dt
    multiplication_2_tab(d_tab_dy,d_tab_dt,-1,d_tab_dy_dt)
    somme_fenetre_global[tuple(gridSize), tuple(blockSize)](d_tab_dy_dt,rayon,d_somme_tab_dy_dt)
    

    #calculer AtA
    d_AtA = np.array([[d_somme_tab_dx_2,d_somme_tab_dx_dy],[d_somme_tab_dx_dy,d_somme_tab_dy_2]])
    #calculer AtA
    d_AtB = np.array([d_somme_tab_dx_dt,d_somme_tab_dy_dt])
    #calculer inv_AtA
    
    inverser_la_matrice(d_AtA,d_premier,d_deuxieme,d_inv_tab)
    

   #calculer dx 
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_inv_AtA[0,0],d_Atb[0],1.,d_dx1)
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_inv_AtA[0,1],d_Atb[1],1.,d_dx2)
    addition_2_tab_GPU[list(gridSize), list(BlockSize)](d_dx1,d_dx2,d_dx)

    #calculer dy
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_inv_AtA[1,0],d_Atb[0],1.,d_dy1)
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_inv_AtA[1,1],d_Atb[1],1.,d_dy2)
    addition_2_tab_GPU[list(gridSize), list(BlockSize)](d_dy1,d_dy2,d_dy)

    dx = d_dx.copy_to_host()
    dy = d_dy.copy_to_host()

    
    
    return dx,dy

    




def main():


    image1 = np.asarray(Image.open("/home/nvidia/iut-optical-flow/data/RubberWhale/frame10.png"))
    image2 = np.asarray(Image.open("/home/nvidia/iut-optical-flow/data/RubberWhale/frame11.png"))
    d_image1 = cu.to_device(image1)
    d_image2 = cu.to_device(image2)
    
    flot_optique(d_image1,d_image2,17)
    
    flowpy.show_flow(dx,dy)
    

if __name__ =="__main__":
    main()
