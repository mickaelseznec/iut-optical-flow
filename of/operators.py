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
 