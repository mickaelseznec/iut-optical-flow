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
    for i in range (hauteur):
        for j in range(largeur):
            somme_tab = np.sum(nouvelle_image[i:i+(2*r)+1,j:j+(2*r)+1])
            somme_fenetre[i][j] = somme_tab
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
