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
    tab_fenetre = np.zeros((2*r+1,2*r+1))
    nouvelle_image = np.zeros((hauteur + 2*r, largeur + 2*r))
    nouvelle_image[r:hauteur+r,r:largeur+r] = image #milieu
    nouvelle_image[0:r,r:hauteur+r] = image[0,:]    #haut
    nouvelle_image[hauteur+r:,r:r+largeur] = image[hauteur-1,:]  #bas
    nouvelle_image[0:r,0:r] = image[0,0] #coin gauche_haut
    nouvelle_image[r+hauteur:,0:r] = image[hauteur-1,0] # coin gauche_bas
    nouvelle_image[0:r,r+largeur:] = image[0,largeur-1]  #coin droit_haut
    nouvelle_image[r+hauteur:,r+largeur:] = image[hauteur-1,largeur-1] #coin droit_bas
    nouvelle_image[r:r+hauteur,0:r]  = np.repeat(image[0:hauteur,0],r).reshape(hauteur,r)
    nouvelle_image[r:r+hauteur,r+hauteur:] = np.repeat(image[0:hauteur,largeur-1],r).reshape(hauteur,r)
    tab_fenetre = nouvelle_image[x:x+2*r+1,y:y+2*r+1]
    som = np.sum(tab_fenetre)
    return som

def flux_optique(image):
    hauteur, largeur = image.shape
    somme_tab_dx_2 = np.zeros((hauteur,largeur))
    somme_tab_dy_2 = np.zeros((hauteur,largeur))
    somme_tab_dx_dy = np.zeros((hauteur,largeur))
    tab_dx_2 = derivee_x(image)*derivee_x(image)
    tab_dy_2 = derivee_y(image)*derivee_y(image)
    tab_dx_dy = derivee_y(image)*derivee_x(image)
    for i in range(hauteur):
        for j in range(largeur):
            somme_tab_dx_2[i,j] = somme_fenetre(tab_dx_2,i,j,1)
            somme_tab_dy_2[i,j] = somme_fenetre(tab_dy_2,i,j,1)
            somme_tab_dx_dy[i,j] = somme_fenetre(tab_dx_dy,i,j,1)
    AtA = np.array([[somme_tab_dx_2,somme_tab_dx_dy],[somme_tab_dx_dy,somme_tab_dy_2]])
    print(AtA)


def inverser_la_matrice(matrice):
    tab_inv = np.zeros((2,2))
    determinant =  matrice[0,0] * matrice[1,1] - matrice[0,1] * matrice[1,0]
    tab_inv[0,0] = matrice[1,1]/determinant
    tab_inv[0,1] = -matrice[0,1]/determinant
    tab_inv[1,0] = -matrice[1,0]/determinant
    tab_inv[1,1] = matrice[0,0]/determinant
    return tab_inv



