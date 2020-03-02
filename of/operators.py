import numpy as np
import scipy.signal
import cv2 as cv
import matplotlib.pyplot as plt
import flowpy
import numba.cuda as cu

    #--------------------------------------------------- Sur CPU ---------------------------------------------------#

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
    tab_inv = matrice
    determinant =  matrice[0,0] * matrice[1,1] - matrice[0,1] * matrice[1,0]
    # print(determinant)
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
    image_1_color = vid.read()[1]
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
def somme_fenetre_global_GPU(d_image,d_r,d_som_tab):
    hauteur,largeur = d_image.shape
    #création de fenêtre
    h,l = cu.grid(2)
    result = 0
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
            result = result + d_image[indice_h, indice_l]
    d_som_tab[h,l] = result 

    
# @cu.jit()
# def flot_optique(d_image1,d_image2,d_rayon,d_dx,d_dy):
#     #création des tableau
#     somme_tab_dx_2 = np.zeros_like(image1)
#     somme_tab_dy_2 = np.zeros_like(image1)
#     somme_tab_dx_dy = np.zeros_like(image1)
#     somme_tab_dx_dt = np.zeros_like(image1)
#     #calculer somme_tab_dx_2
#     blockSize = np.array([32, 32])
#     gridSize = (np.asarray(array_1.shape) + (blockSize - 1)) // blockSize

#     derivee_x[tuple(gridSize), tuple(blockSize)](d_image_1,d_tab_dx)
#     tab_dx_2 = d_tab_dx * d_tab_dx
#     somme_fenetre_gobal[tuple(gridSize), tuple(blockSize)](tab_dx_2,d_rayon,d_somme_tab_dx_2)
    
#     #calculer somme_tab_dy_2
#     derivee_y[tuple(gridSize), tuple(blockSize)](d_image_1,d_tab_dy)
#     tab_dy_2 = d_tab_dy * d_tab_dy
#     somme_fenetre_gobal[tuple(gridSize), tuple(blockSize)](tab_dy_2,d_rayon,d_somme_tab_dy_2)
    

#     #calculer somme_tab_dx_dy
#     tab_dx_dy = d_tab_dx * d_tab_dy
#     somme_fenetre_gobal[tuple(gridSize), tuple(blockSize)](tab_dx_dy,d_rayon,d_somme_tab_dx_dy)
   

#     #calculer somme_tab_dx_dt
#     derivee_t[tuple(gridSize), tuple(blockSize)](d_image_1,d_image_2,d_tab_dt)
#     tab_dx_dt = d_tab_dx * -d_tab_dt
#     somme_fenetre_gobal[tuple(gridSize), tuple(blockSize)](tab_dx_dt,d_rayon,d_somme_tab_dx_dt)
    

#     #calculer somme_tab_dy_dt
#     tab_dy_dt = tab_dy * -d_tab_dt
#     somme_fenetre_gobal[tuple(gridSize), tuple(blockSize)](tab_dy_dt,d_rayon,d_somme_tab_dy_dt)
    

#     #calculer AtA
#     AtA = np.array([[d_somme_tab_dx_2,d_somme_tab_dx_dy],[d_somme_tab_dx_dy,d_somme_tab_dy_2]])
#     #calculer AtB
#     AtB = np.array([d_somme_tab_dx_dt,d_somme_tab_dy_dt])
#     #calculer inv_AtA
#     inverser_la_matrice[tuple(gridSize), tuple(blockSize)](d_matrice,d_inv_AtA)
    

#     #calculer dx 
#     d_dx = d_inv_AtA[0][0] * AtB[0] + d_inv_AtA[0][1] * AtB[1]
#     d_dy = d_inv_AtA[1][0] * AtB[0] + d_inv_AtA[1][1] * AtB[1]
   
# @cu.jit()
# def inverser_la_matrice(d_matrice,d_tab_inv):
#     d_tab_inv = d_matrice
#     determinant =  d_matrice[0,0] * d_matrice[1,1] - d_matrice[0,1] * d_matrice[1,0]
#     d_dtab_inv[0,0] = d_matrice[1,1]/determinant
#     d_tab_inv[0,1] = -1*d_matrice[0,1]/determinant
#     d_tab_inv[1,0] = -1*d_matrice[1,0]/determinant
#     d_tab_inv[1,1] = d_matrice[0,0]/determinant
    




# def main():
   
#     image1 = np.asarray(Image.open("data/RubberWhale/frame10.png"))
#     image2 = np.asarray(Image.open("data/RubberWhale/frame11.png"))

#     d_image_1 = cu.to_device(image_1)
#     d_image_2 = cu.to_device(image_2)
#     d_r = cu.to_device(r)

#     blockSize = np.array([32, 32])
#     gridSize = (np.asarray(array_1.shape) + (blockSize - 1)) // blockSize

#     flot_optique[tuple(gridSize), tuple(blockSize)](d_image_1, d_image_2,d_r,d_dx,d_dy)
#     image_dx = d_dx.copy_to_host()
#     image_dy = d_dy.copy_to_host()
#     flowpy.show_flow(image_dx,image_dy)