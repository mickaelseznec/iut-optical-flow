import numpy as np
import scipy.signal
import cv2 as cv
import matplotlib.pyplot as plt
import flowpy
import numba
import numba.cuda as cu

from scipy.ndimage import map_coordinates

    #--------------------------------------------------- Sur CPU ---------------------------------------------------#
def warp(image, flow):
    height, width = image.shape
    coord = np.mgrid[:height,:width]

    points = coord.transpose(1,2,0).reshape((-1,2))
    gx = (coord[1] + flow[0])
    gy = (coord[0] + flow[1])

    return map_coordinates(image, (gy, gx), mode="nearest")

@numba.jit()
def warp_parallel(image, flow_y, flow_x):
    height, width = image.shape

    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            y_shift = y + flow_y[y, x]
            x_shift = x + flow_x[y, x]

            x_shift = max(min(width - 1, x_shift), 0)
            y_shift = max(min(height - 1, y_shift), 0)

            lower_x, lower_y = int(x_shift), int(y_shift)
            upper_x, upper_y = lower_x + 1, lower_y + 1

            residual_x = x_shift - lower_x
            residual_y = y_shift - lower_y

            result[y, x] = (image[lower_y, lower_x] * (1 - residual_y) * (1 - residual_x) +
                            image[lower_y, upper_x] * (1 - residual_y) * residual_x +
                            image[upper_y, lower_x] * residual_y * (1 - residual_x) +
                            image[upper_y, upper_x] * residual_y * residual_x)

    return result

def derivee_x(image):
    # Retourne la derivée en x pour tous les pixels de l'image
    tab_dx = np.zeros_like(image)
    tab_dx[:,-1] = 0
    tab_dx[:,0:-1] =  image[:,1:] - image[:,0:-1]
    return tab_dx

def derivee_y(image):
    # Retourne la derivée en y pour tous les pixels de l'image
    tab_dy = np.zeros_like(image)
    tab_dy[-1,:] = 0
    tab_dy[0:-1,:] =  image[1:,:] - image[0:-1,:]   
    return tab_dy

def derivee_t(image_1, image_2):
    # Retourne la derivée en t pour tous les pixels de l'image
    tab_dt = np.zeros_like(image_1)
    tab_dt  =  image_2.astype(float) - image_1.astype(float)
    return tab_dt
 
def somme_fenetre(image,x,y,r):
    hauteur,largeur = image.shape
    # Création de la nouvelle image
    nouvelle_image = np.zeros((2*r+hauteur,2*r+largeur))
    nouvelle_image[r:hauteur+r,r:largeur+r] = image                                                                                 #Milieu du tableau
    nouvelle_image[0:r,r:largeur+r] = image[0,:]                                                                                    #Haut du tableau
    nouvelle_image[r+hauteur:,r:largeur+r] = image[hauteur-1:hauteur,:]                                                             #Bas du tableau
    nouvelle_image[0:r,0:r] = image[0][0]                                                                                           #coin haut gauche du tableau
    nouvelle_image[0:r,largeur+r:] = image[0][largeur-1]                                                                            #coin haut droite du tableau
    nouvelle_image[hauteur+r:,largeur+r:] = image[hauteur-1][largeur-1]                                                             #coin bas droite du tableau
    nouvelle_image[hauteur+r:,0:r] = image[hauteur-1][0]                                                                            #coin bas gauche du tableau
    nouvelle_image[r:hauteur+r,0:r] = np.repeat(image[:hauteur,0:1],r).reshape(hauteur,r)                                           #gauche du tableau
    nouvelle_image[r:hauteur+r,largeur+r:largeur+2*r] = np.repeat(image[0:hauteur,largeur-1:largeur],r).reshape(hauteur,r)          #droite du tableau
    # Création de la fenêtre
    tab_fenetre = np.zeros((2*r+1,2*r+1))
    tab_fenetre = nouvelle_image[x:x+(2*r)+1,y:y+(2*r)+1]
    # Calcul de la somme du tableau
    somme_tab = np.sum(tab_fenetre)
    return somme_tab

def somme_fenetre_global(image,r):
    hauteur,largeur = image.shape
    # Création de la nouvelle image
    nouvelle_image = np.zeros((2*r+hauteur,2*r+largeur))
    nouvelle_image[r:hauteur+r,r:largeur+r] = image                                                                                 #Milieu du tableau
    nouvelle_image[0:r,r:largeur+r] = image[0,:]                                                                                    #Haut du tableau
    nouvelle_image[r+hauteur:,r:largeur+r] = image[hauteur-1:hauteur,:]                                                             #Bas du tableau
    nouvelle_image[0:r,0:r] = image[0][0]                                                                                           #coin haut gauche du tableau
    nouvelle_image[0:r,largeur+r:] = image[0][largeur-1]                                                                            #coin haut droite du tableau
    nouvelle_image[hauteur+r:,largeur+r:] = image[hauteur-1][largeur-1]                                                             #coin bas droite du tableau
    nouvelle_image[hauteur+r:,0:r] = image[hauteur-1][0]                                                                            #coin bas gauche du tableau
    nouvelle_image[r:hauteur+r,0:r] = np.repeat(image[:hauteur,0:1],r).reshape(hauteur,r)                                           #gauche du tableau
    nouvelle_image[r:hauteur+r,largeur+r:largeur+2*r] = np.repeat(image[0:hauteur,largeur-1:largeur],r).reshape(hauteur,r)          #droite du tableau
    # Création du tableau final
    somme_fenetre = np.zeros((hauteur,largeur))
    # Calcul de la somme du tableau
    filtre = np.ones((2*r+1,2*r+1))
    somme_image_intermediaire = scipy.signal.convolve2d(nouvelle_image,filtre)
    somme_fenetre = somme_image_intermediaire[r+r:-(r+r),r+r:-(r+r)]
    return somme_fenetre

def inverser_matrice(matrice):
    tab_inv = np.zeros_like(matrice)
    determinant =  matrice[0,0] * matrice[1,1] - matrice[0,1] * matrice[1,0]
    # print(matrice[0,0] * matrice[1,1])
    tab_inv[0,0] = matrice[1,1]
    tab_inv[0,1] = -1*matrice[0,1]
    tab_inv[1,0] = -1*matrice[1,0]
    tab_inv[1,1] = matrice[0,0]
    return tab_inv/determinant

def flux_optique(image1, image2, rayon):
    # Création de la forme des tableaux
    somme_tab_dx_2 = np.zeros_like(image1)
    somme_tab_dy_2 = np.zeros_like(image1)
    somme_tab_dx_dy = np.zeros_like(image1)
    somme_dt_dx = np.zeros_like(image1)
    somme_dt_dy = np.zeros_like(image1)
    # Création des différents tableaux
    tab_dx_2 = derivee_x(image1)*derivee_x(image1)
    tab_dy_2 = derivee_y(image1)*derivee_y(image1)
    tab_dx_dy = derivee_y(image1)*derivee_x(image1)
    tab_dt = -derivee_t(image1,image2)
    tab_dt_dx = tab_dt*derivee_x(image1)
    tab_dt_dy = tab_dt*derivee_y(image1)
    # Création des tableaux sommés
    somme_tab_dx_2 = somme_fenetre_global(tab_dx_2,rayon)
    somme_tab_dy_2 = somme_fenetre_global(tab_dy_2,rayon)
    somme_tab_dx_dy = somme_fenetre_global(tab_dx_dy,rayon)
    somme_dt_dx = somme_fenetre_global(tab_dt_dx,rayon)
    somme_dt_dy = somme_fenetre_global(tab_dt_dy,rayon)
    # Mise en place des matrices
    AtA = np.array([[somme_tab_dx_2,somme_tab_dx_dy],[somme_tab_dx_dy,somme_tab_dy_2]])
    inv_AtA = inverser_matrice(AtA)
    Atb = np.array([somme_dt_dx,somme_dt_dy])
    # Valeurs de sortie
    dx = inv_AtA[0][0]*Atb[0] + inv_AtA[0][1]*Atb[1]
    dy = inv_AtA[1][0]*Atb[0] + inv_AtA[1][1]*Atb[1]
    return dx, dy

def flux_optique_video(video, rayon):
    vid = cv.VideoCapture(video)
    ret, image_1_color = vid.read()
    image_1 = cv.cvtColor(np.array(image_1_color), cv.COLOR_BGR2GRAY)
    hauteur, largeur = image_1.shape
    image_1 = cv.resize(image_1,(largeur//4,hauteur//4))
    image_1_color = cv.resize(image_1_color,(largeur//4,hauteur//4))
    # plt.imshow(image_1)
    # plt.show()

    while(vid.isOpened()):
        ret, image_2_color = vid.read()
        if not ret:
            break
        image_2 = cv.cvtColor(np.array(image_2_color), cv.COLOR_BGR2GRAY)
        image_2 = cv.resize(image_2,(largeur//4,hauteur//4))
        image_2_color = cv.resize(image_2_color,(largeur//4,hauteur//4))
        if cv.waitKey(500) & 0xFF == ord('q'):
            break
        x, y = flux_optique(image_1, image_2, rayon)
        img = flowpy.flow_to_color(x,y, max_norm=5)
        # norme = np.sqrt(x**2+y**2)
        output = cv.add(image_1_color, img)
        cv.imshow("sparse optical flow", output)
        image_1 = image_2
        image_1_color = image_2_color
    vid.release()
    cv.destroyAllWindows()

    #--------------------------------------------------- Sur GPU ---------------------------------------------------#

@cu.jit()
def derivee_x_GPU(d_image,d_tab_dx):
    # Retourne la derivée en x pour tous les pixels de l'image
    h, l = d_image.shape
    y,x = cu.grid(2)
    if x >= l or y >= h:
        return
    if x == l-1 :
        d_tab_dx[y,x] = 0
    else:
        d_tab_dx[y,x] = d_image[y,x+1] - d_image[y,x]     
    

@cu.jit()
def derivee_y_GPU(d_image,d_tab_dy):
    # Retourne la derivée en y pour tous les pixels de l'image
    h, l = d_image.shape
    y,x = cu.grid(2)
    if x >= l or y >= h:
        return
    if y == h-1:
        d_tab_dy[y,x] = 0
    else:
        d_tab_dy[y,x] = d_image[y+1,x] - d_image[y,x]  
   

@cu.jit()
def derivee_t_GPU(d_image_1, d_image_2,d_tab_dt):
    # Retourne la derivée en t pour tous les pixels de l'image
    h, l = cu.grid(2)
    d_tab_dt[h,l] =  d_image_2[h,l] - d_image_1[h,l]


@cu.jit()
def multiplication_2_tab_GPU(d_tab_1, d_tab_2, d_coef, d_tab_mult):
    hauteur, largeur = d_tab_1.shape
    h,l = cu.grid(2)
    if (h >= hauteur or l >= largeur):
        return
    d_tab_mult[h,l] = d_tab_1[h,l]*d_coef*d_tab_2[h,l]


@cu.jit()
def division_2_tab_GPU(d_numerateur, d_denominateur, d_coef, d_tab_sortie):
    hauteur, largeur = d_numerateur.shape
    h,l = cu.grid(2)
    if (h >= hauteur or l >= largeur):
        return
    d_tab_sortie[h,l] = d_coef*d_numerateur[h,l]/d_denominateur[h,l]

@cu.jit()
def soustraction_2_tab_GPU(d_tab1, d_tab2, d_tab_sortie):
    hauteur, largeur = d_tab1.shape
    h,l = cu.grid(2)
    if (h >= hauteur or l >= largeur):
        return
    d_tab_sortie[h,l] = d_tab1[h,l]-d_tab2[h,l]

@cu.jit()
def addition_2_tab_GPU(d_tab1, d_tab2, d_tab_sortie):
    hauteur, largeur = d_tab1.shape
    h,l = cu.grid(2)
    if (h >= hauteur or l >= largeur):
        return
    d_tab_sortie[h,l] = d_tab1[h,l]+d_tab2[h,l]
    

def inverser_la_matrice_GPU(d_matrice, d_premier, d_deuxieme, d_determinant, d_matrice_inverse):
    BlockSize = np.array([32,32])
    gridSize = (np.asarray(d_matrice[0,0].shape) + (BlockSize-1))//BlockSize

    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_matrice[0,0], d_matrice[1,1], 1., d_premier) 
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_matrice[0,1], d_matrice[1,0], 1., d_deuxieme)
    
    
    soustraction_2_tab_GPU[list(gridSize), list(BlockSize)](d_premier, d_deuxieme, d_determinant)


    division_2_tab_GPU[list(gridSize), list(BlockSize)](d_matrice[1,1],d_determinant,1.,d_matrice_inverse[0,0])
    division_2_tab_GPU[list(gridSize), list(BlockSize)](d_matrice[0,1],d_determinant,-1.,d_matrice_inverse[0,1])
    division_2_tab_GPU[list(gridSize), list(BlockSize)](d_matrice[1,0],d_determinant,-1.,d_matrice_inverse[1,0])
    division_2_tab_GPU[list(gridSize), list(BlockSize)](d_matrice[0,0],d_determinant,1.,d_matrice_inverse[1,1]) 


@cu.jit()
def somme_fenetre_global_GPU(d_image,d_r,d_som_tab):
    hauteur,largeur = d_image.shape  
    #création de fenêtre
    h,l = cu.grid(2) 
    result= 0
    if (h >= hauteur or l >= largeur):
        return
    for i in range((d_r*2)+1):
        for j in range((d_r*2)+1):
            indice_l = l-d_r + j
            indice_h = h-d_r + i
            if (indice_h < 0): 
                indice_h = 0
            if (indice_l < 0):
                indice_l = 0
            if (indice_l >= largeur):
                indice_l = largeur-1
            if (indice_h >= hauteur):
                indice_h = hauteur-1
            result += d_image[indice_h, indice_l]
    d_som_tab[h,l] = result


def flux_optique_GPU(d_image1, d_image2, d_rayon,d_matrice22, d_matrice11):
    BlockSize = np.array([32,32])
    gridSize = (np.asarray(d_image1.shape) + (BlockSize-1))//BlockSize
    #création des tableau
    d_tab_dx = cu.device_array_like(d_image1)
    d_tab_dx_2 = cu.device_array_like(d_image1)
    d_somme_tab_dx_2 = cu.device_array_like(d_image1)
    d_tab_dy = cu.device_array_like(d_image1)
    d_tab_dy_2 = cu.device_array_like(d_image1)
    d_somme_tab_dy_2 = cu.device_array_like(d_image1)
    d_tab_dx_dy = cu.device_array_like(d_image1)
    d_somme_tab_dx_dy = cu.device_array_like(d_image1)
    d_tab_dt = cu.device_array_like(d_image1)
    d_tab_dx_dt = cu.device_array_like(d_image1)
    d_somme_tab_dx_dt = cu.device_array_like(d_image1)
    d_tab_dy_dt = cu.device_array_like(d_image1)
    d_somme_tab_dy_dt = cu.device_array_like(d_image1)
    d_AtA = cu.device_array_like(d_matrice22)
    d_Atb = cu.device_array_like(d_matrice11)
    d_inv_AtA = cu.device_array_like(d_matrice22)
    d_premier = cu.device_array_like(d_image1)
    d_deuxieme = cu.device_array_like(d_image1)
    d_determinant = cu.device_array_like(d_image1)
    d_dx = cu.device_array_like(d_image1)
    d_dx1 = cu.device_array_like(d_image1)
    d_dx2 = cu.device_array_like(d_image1)
    d_dy = cu.device_array_like(d_image1)
    d_dy1 = cu.device_array_like(d_image1)
    d_dy2 = cu.device_array_like(d_image1)
    

    #calculer somme_tab_dx_2
    derivee_x_GPU[list(gridSize), list(BlockSize)](d_image1, d_tab_dx)
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_tab_dx, d_tab_dx, 1., d_tab_dx_2)
    somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](d_tab_dx_2, d_rayon, d_somme_tab_dx_2)
    
    #calculer somme_tab_dy_2
    derivee_y_GPU[list(gridSize), list(BlockSize)](d_image1, d_tab_dy)
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_tab_dy, d_tab_dy, 1., d_tab_dy_2)
    somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](d_tab_dy_2, d_rayon, d_somme_tab_dy_2)
    
    #calculer somme_tab_dx_dy
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_tab_dx, d_tab_dy, 1., d_tab_dx_dy)
    somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](d_tab_dx_dy,d_rayon,d_somme_tab_dx_dy)
   
    #calculer somme_tab_dx_dt
    derivee_t_GPU[list(gridSize), list(BlockSize)](d_image1,d_image2,d_tab_dt)
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_tab_dx, d_tab_dt, -1., d_tab_dx_dt)
    somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](d_tab_dx_dt,d_rayon,d_somme_tab_dx_dt)
    
    #calculer somme_tab_dy_dt
    multiplication_2_tab_GPU[list(gridSize), list(BlockSize)](d_tab_dy, d_tab_dt, -1., d_tab_dy_dt)
    somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](d_tab_dy_dt,d_rayon,d_somme_tab_dy_dt)
    
    #calculer AtA
    # AtA = np.array([[d_somme_tab_dx_2,d_somme_tab_dx_dy],[d_somme_tab_dx_dy,d_somme_tab_dy_2]])
    d_AtA[0,0] = d_somme_tab_dx_2
    d_AtA[0,1] = d_somme_tab_dx_dy
    d_AtA[1,0] = d_somme_tab_dx_dy
    d_AtA[1,1] = d_somme_tab_dy_2

    #calculer AtB
    d_Atb[0] = d_somme_tab_dx_dt
    d_Atb[1] = d_somme_tab_dy_dt
    # AtB = np.array([d_somme_tab_dx_dt,d_somme_tab_dy_dt])

    #calculer inv_AtA
    # inverser_la_matrice_GPU[list(gridSize), list(BlockSize)](d_AtA,d_inv_AtA)
    inverser_la_matrice_GPU(d_AtA, d_premier, d_deuxieme, d_determinant, d_inv_AtA)
    

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

    return dx, dy   