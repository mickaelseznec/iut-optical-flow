import numpy as np


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
    tab_dt  =  image_2 - image_1
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

def inverser_matrice(matrice):
    tab_inv = matrice
    determinant =  matrice[0,0] * matrice[1,1] - matrice[0,1] * matrice[1,0]
    # print(determinant)
    tab_inv[0,0] = matrice[1,1]
    tab_inv[0,1] = -1*matrice[0,1]
    tab_inv[1,0] = -1*matrice[1,0]
    tab_inv[1,1] = matrice[0,0]
    return tab_inv/determinant

def flux_optique(image1,image2):
    hauteur, largeur = image1.shape
    somme_tab_dx_2 = np.zeros_like(image1)
    somme_tab_dy_2 = np.zeros_like(image1)
    somme_tab_dx_dy = np.zeros_like(image1)
    somme_dt_dx = np.zeros_like(image1)
    somme_dt_dy = np.zeros_like(image1)
    tab_dx_2 = derivee_x(image1)*derivee_x(image1)
    tab_dy_2 = derivee_y(image1)*derivee_y(image1)
    tab_dx_dy = derivee_y(image1)*derivee_x(image1)
    b = -derivee_t(image1,image2)
    tab_dt_dx = b*derivee_x(image1)
    tab_dt_dy = b*derivee_y(image1)
    for i in range(hauteur):
        for j in range(largeur):
            somme_tab_dx_2[i][j] = somme_fenetre(tab_dx_2,i,j,1)
            somme_tab_dy_2[i][j] = somme_fenetre(tab_dy_2,i,j,1)
            somme_tab_dx_dy[i][j] = somme_fenetre(tab_dx_dy,i,j,1)
            somme_dt_dx[i][j] = somme_fenetre(tab_dt_dx,i,j,1)
            somme_dt_dy[i][j] = somme_fenetre(tab_dt_dy,i,j,1)
    AtA = np.array([[somme_tab_dx_2,somme_tab_dx_dy],[somme_tab_dx_dy,somme_tab_dy_2]])
    inv_AtA = inverser_matrice(AtA)
    Atb = np.array([somme_dt_dx,somme_dt_dy])
    dx = AtA[0][0]*Atb[0] + AtA[0][1]*Atb[1]
    dy = AtA[1][0]*Atb[0] + AtA[1][1]*Atb[1]
    print(dy)
