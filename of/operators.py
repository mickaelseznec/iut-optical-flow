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
 
def somme_fenetre(image,x,y,r):  # x correspond à ligne, y correspond à colone
    hauteur,largeur = image.shape
    #création de fenêtre
    tab_fenetre = np.zeros((2*r+1,2*r+1))
    nouvelle_image = np.zeros((hauteur + 2*r, largeur + 2*r))
    nouvelle_image[r:hauteur+r,r:largeur+r] = image #milieu
    nouvelle_image[0:r,r:largeur+r] = image[0,:]    #haut
    nouvelle_image[hauteur+r:,r:r+largeur] = image[hauteur-1,:]  #bas
    nouvelle_image[0:r,0:r] = image[0,0] #coin gauche_haut
    nouvelle_image[r+hauteur:,0:r] = image[hauteur-1,0] # coin gauche_bas
    nouvelle_image[0:r,r+largeur:] = image[0,largeur-1]  #coin droit_haut
    nouvelle_image[r+hauteur:,r+largeur:] = image[hauteur-1,largeur-1] #coin droit_bas
    nouvelle_image[r:r+hauteur,0:r]  = np.repeat(image[0:hauteur,0],r).reshape(hauteur,r)
    nouvelle_image[r:r+hauteur,r+largeur:] = np.repeat(image[0:hauteur,largeur-1],r).reshape(hauteur,r)
    #calculer la somme de tableau
    tab_fenetre = nouvelle_image[x:x+2*r+1,y:y+2*r+1]
    som = np.sum(tab_fenetre)
    return som

def somme_fenetre_global(image,r):
    hauteur,largeur = image.shape
    #création de fenêtre
    tab_fenetre = np.zeros((2*r+1,2*r+1))
    som_tab = np.zeros((hauteur,largeur))
    nouvelle_image = np.zeros((hauteur + 2*r, largeur + 2*r))
    nouvelle_image[r:hauteur+r,r:largeur+r] = image #milieu
    nouvelle_image[0:r,r:largeur+r] = image[0,:]    #haut
    nouvelle_image[hauteur+r:,r:r+largeur] = image[hauteur-1,:]  #bas
    nouvelle_image[0:r,0:r] = image[0,0] #coin gauche_haut
    nouvelle_image[r+hauteur:,0:r] = image[hauteur-1,0] # coin gauche_bas
    nouvelle_image[0:r,r+largeur:] = image[0,largeur-1]  #coin droit_haut
    nouvelle_image[r+hauteur:,r+largeur:] = image[hauteur-1,largeur-1] #coin droit_bas
    nouvelle_image[r:r+hauteur,0:r]  = np.repeat(image[0:hauteur,0],r).reshape(hauteur,r)
    nouvelle_image[r:r+hauteur,r+largeur:] = np.repeat(image[0:hauteur,largeur-1],r).reshape(hauteur,r)
    for i in range(hauteur):
        for j in range(largeur):
            tab_fenetre = nouvelle_image[i:i+2*r+1,j:j+2*r+1] 
            som_tab[i][j] = np.sum(tab_fenetre)
    return som_tab

def flot_optique(image1,image2,rayon):
    #création des tableau
    somme_tab_dx_2 = np.zeros_like(image1)
    somme_tab_dy_2 = np.zeros_like(image1)
    somme_tab_dx_dy = np.zeros_like(image1)
    somme_tab_dx_dt = np.zeros_like(image1)
    #calculer somme_tab_dx_2
    tab_dx =  derivee_x(image1)
    tab_dx_2 = tab_dx * tab_dx
    somme_tab_dx_2 = somme_fenetre_global(tab_dx_2, rayon)
    #calculer somme_tab_dy_2
    tab_dy =  derivee_y(image1)
    tab_dy_2 = tab_dy * tab_dy
    somme_tab_dy_2 = somme_fenetre_global(tab_dy_2, rayon)
    #calculer somme_tab_dx_dy
    tab_dx_dy = tab_dx * tab_dy
    somme_tab_dx_dy = somme_fenetre_global(tab_dx_dy,rayon)
    #calculer somme_tab_dx_dt
    tab_dt =  derivee_t(image1,image2)
    tab_dx_dt = tab_dx * tab_dt
    somme_tab_dx_dt = somme_fenetre_global(tab_dx_dt,rayon)
    #calculer somme_tab_dy_dt
    tab_dy_dt = tab_dy * tab_dt
    somme_tab_dy_dt = somme_fenetre_global(tab_dy_dt,rayon)
    #calculer AtA
    AtA = np.array([[somme_tab_dx_2,somme_tab_dx_dy],[somme_tab_dx_dy,somme_tab_dy_2]])
    #calculer AtB
    AtB = np.array([somme_tab_dx_dt,somme_tab_dy_dt])
    #calculer inv_AtA
    inv_AtA = inverser_la_matrice(AtA)

    #calculer dx 
    dx = inv_AtA[0][0] * AtB[0] + inv_AtA[0][1] * AtB[1]
    dy = inv_AtA[1][0] * AtB[0] + inv_AtA[1][1] * AtB[1]
    return dx,dy

def inverser_la_matrice(matrice):
    tab_inv = matrice
    determinant =  matrice[0,0] * matrice[1,1] - matrice[0,1] * matrice[1,0]
    tab_inv[0,0] = matrice[1,1]
    tab_inv[0,1] = -1*matrice[0,1]
    tab_inv[1,0] = -1*matrice[1,0]
    tab_inv[1,1] = matrice[0,0]
    return tab_inv/determinant



