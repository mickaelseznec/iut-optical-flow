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
            tab_int = image[0:x+r+1,y-r:largeur]
            tab_bord1 = image[0,y-r:largeur]
            tab_bord2 = image[0:x+r+1,largeur-1]
            som= np.sum(tab_int) + np.sum(tab_bord1) * (r-x) + np.sum(tab_bord2) * (y+r-largeur+1) + image[0,largeur-1] * (r-x) * (y+r-largeur+1)
            print(tab_int)
            print(tab_bord1)
            print(tab_bord2)
        elif (y-r<0):
            tab_int = image[0:x+r+1,0:y+r+1]
            tab_bord1 = image[0,0:y+r+1]
            tab_bord2 = image[0:x+r+1,0]
            som= np.sum(tab_int) + np.sum(tab_bord1) * (r-x) + np.sum(tab_bord2) * (r-y) + image[0,0] * (r-x) * (r-y) 
        else:
            tab_int = image[0:x+r+1,y-r:y+r+1]
            tab_bord= image[0,y-r:y+r+1]
            som = np.sum(tab_int) + np.sum(tab_bord) * (r-x)

    
    
    
    elif(x+r>(hauteur-1)):
        if(y+r>largeur-1):
            if(y+r>largeur-1):
                tab_int = image[x-r:hauteur,y-r:largeur]
                tab_bord1 = image[hauteur-1,y-r:largeur]
                tab_bord2 = image[x-r:hauteur,largeur-1]
                som= np.sum(tab_int) + np.sum(tab_bord1) * (x+r-hauteur+1) + np.sum(tab_bord2) * (y+r-largeur+1) + image[hauteur-1,largeur-1] * (x+r-hauteur+1) * (y+r-largeur+1)
                print(tab_int)
                print(tab_bord1)
                print(tab_bord2)
        elif (y-r<0):
            tab_int = image[x-r:hauteur,0:y+r+1]
            tab_bord1 = image[hauteur-1,0:y+r+1]
            tab_bord2 = image[x-r:hauteur,0]
            som= np.sum(tab_int) + np.sum(tab_bord1) * (x+r-hauteur+1) + np.sum(tab_bord2) * (r-y) + image[hauteur-1,0] * (x+r-hauteur+1) * (r-y)
            print(tab_int)
            print(tab_bord1)
            print(tab_bord2)
        else:
            tab_int = image[x-r:hauteur,y-r:y+r+1]
            tab_bord= image[(hauteur-1),y-r:y+r+1]
            som = np.sum(tab_int) + np.sum(tab_bord) * (r-(hauteur-1-x)) 
    
    
    
    else:
        if(y+r>largeur-1):
            tab_int = image[x-r:x+r+1,y-r:largeur]
            tab_bord= image[x-r:x+r+1,largeur-1]          
            som = np.sum(tab_int) + np.sum(tab_bord) * (y+r-largeur+1)
        elif (y-r<0):
            tab_int = image[x-r:x+r+1,0:y+r+1]
            tab_bord= image[x-r:x+r+1,0]          
            som = np.sum(tab_int) + np.sum(tab_bord) * (r-y)
    return som
