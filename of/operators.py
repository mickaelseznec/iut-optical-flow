import numpy as np


def derivee_x(image):

    # Retourne la derivée en x pour tous les pixels de l'image
    hauteur, largeur = image.shape
    tab_dx = np.zeros((hauteur,largeur))
    tab_dx[:,-1] = 0
    tab_dx[:,0:-1] =  image[:,1:] - image[:,0:-1]    
    return tab_dx

def derivee_y(image):
    # Retourne la derivée en y pour tous les pixels de l'image
    hauteur, largeur = image.shape
    tab_dy = np.zeros((hauteur,largeur))
    tab_dy[-1,:] = 0
    tab_dy[0:-1,:] =  image[1:,:] - image[0:-1,:]   
    return tab_dy

def derivee_t(image_1, image_2):
    # Retourne la derivée en t pour tous les pixels de l'image
    hauteur,largeur = image_1.shape
    tab_dt = np.zeros((hauteur,largeur))
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
    tab_inv = np.zeros((2,2))
    determinant =  matrice[0,0] * matrice[1,1] - matrice[0,1] * matrice[1,0]
    print(determinant)
    tab_inv[0,0] = matrice[1,1]/determinant
    tab_inv[0,1] = -matrice[0,1]/determinant
    tab_inv[1,0] = -matrice[1,0]/determinant
    tab_inv[1,1] = matrice[0,0]/determinant
    return tab_inv