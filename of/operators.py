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
    somme_tab = 0
    if((x+r < hauteur) &  (x-r >= 0) & (y+r < largeur ) & (y-r >= 0)):
        # pas de dépassement
        tab_fenetre = image[x-r:x+r+1,y-r:y+r+1]
        somme_tab = np.sum(tab_fenetre)
    elif(x-r<0):
        if(y+r>largeur-1):
            # dépassement en coin haut droite
            tab_int = image[0:x+r+1,y-r:largeur]
            tab_bord_haut = image[0,y-r:largeur]
            tab_bord_droite = image[0:x+r+1,largeur-1:largeur]
            coin = image[0][largeur-1]
            somme_tab = np.sum(tab_int) + np.sum(tab_bord_haut)*(r-x) + np.sum(tab_bord_droite)*(r-(largeur-1-y)) + coin*((r-x)*(r-(largeur-1-y)))
        elif (y-r<0):
            # dépassement en coin haut gauche
            tab_int = image[0:x+r+1,0:y+r+1]
            tab_bord_haut = image[0,0:y+r+1]
            tab_bord_gauche = image[0:x+r+1,0:1]
            coin = image[0][0]
            somme_tab = np.sum(tab_int) + np.sum(tab_bord_haut)*(r-x) + np.sum(tab_bord_gauche)*(r-y) + coin*((r-x)*(r-y))
        else:
            # dépassement en haut seulement
            tab_int = image[0:x+r+1,y-r:y+r+1]
            tab_bord= image[0,y-r:y+r+1]
            somme_tab = np.sum(tab_int) + np.sum(tab_bord) * (r-x)

    elif(x+r>(hauteur-1)):
        if(y+r>largeur-1):
            # dépassement en coin bas droite
            tab_int = image[x-r:hauteur,y-r:largeur]
            tab_bord_bas = image[hauteur-1:hauteur,y-r:largeur]
            tab_bord_droite = image[x-r:hauteur,largeur-1:largeur]
            coin = image[hauteur-1][largeur-1]
            somme_tab = np.sum(tab_int) + np.sum(tab_bord_bas)*(r-(hauteur-1-x)) + np.sum(tab_bord_droite)*(r-(largeur-1-y)) + coin*((r-(hauteur-1-x))*(r-(largeur-1-y)))
        elif (y-r<0):
            # dépassement en coin bas gauche
            tab_int = image[x-r:hauteur,0:y+r+1]
            tab_bord_bas = image[(hauteur-1):hauteur,0:y+r+1]
            tab_bord_gauche = image[x-r:hauteur,0:1]
            coin = image[hauteur-1][0]
            somme_tab = np.sum(tab_int) + np.sum(tab_bord_bas)*(r-(hauteur-1-x)) + np.sum(tab_bord_gauche)*(r-y) + coin*((r-(hauteur-1-x))*(r-y))
            pass
        else:
            # dépassement en bas seulement
            tab_int = image[x-r:(hauteur),y-r:y+r+1]
            tab_bord= image[(hauteur-1),y-r:y+r+1]
            somme_tab = np.sum(tab_int) + np.sum(tab_bord) * (r-(hauteur-1-x))

    elif(y-r<0):
        # dépassement à gauche seulement
        tab_int = image[x-r:x+r+1,0:y+r+1]
        tab_bord= image[x-r:x+r+1,0:1]
        somme_tab = np.sum(tab_int) + np.sum(tab_bord) * (r-y)
    
    elif(y+r>largeur-1):
        # dépassement à droite seulement
        tab_int = image[x-r:x+r+1,y-r:largeur]
        tab_bord= image[x-r:x+r+1,(largeur-1):largeur]
        somme_tab = np.sum(tab_int) + np.sum(tab_bord) * (r-(largeur-1-y))
    return somme_tab