# -*- coding: utf-8 -*-

"""
GROUPE DE TD : P

MARCAL THOMAS
MARQUET THOMAS

~~~~~~~~~~~~~~~~~~ KNN - Classification Challenge ~~~~~~~~~~~~~~~~~~

"""


# %% Bibliothèques

import numpy as np
import math as m
import matplotlib.pyplot as plt
import time


# %% Extraction des données

# %% DATA 1 DU DEVOIR KNN

l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,label = [],[],[],[],[],[],[],[],[],[],[]

f = open(r"C:\Users\thoma\Desktop\KNN\data.txt", "r") 
data = f.read()
lines = data.splitlines()
f.close()

l=[]
for i in range(len(lines)):
    l = lines[i].split(';')
    l1.append(l[0])
    l2.append(l[1])
    l3.append(l[2])
    l4.append(l[3])
    l5.append(l[4])
    l6.append(l[5])
    l7.append(l[6])
    l8.append(l[7])
    l9.append(l[8])
    l10.append(l[9])
    label.append(l[10])
    
Data_data = []
for k in range(len(l1)):
    Data_data.append([l1[k], l2[k], l3[k], l4[k], l5[k], l6[k], l7[k], l8[k], l9[k], l10[k], label[k]])
  
"""
# Partie des premiers test réalisés à partir de cette base, pour une première vérification du KNN  
  
random.shuffle(Data_data) # On mélange les échantillons pour avoir un ensembles données homogènes pour l'apprentissage
Apprentissage = Data_data[:int(len(Data_data)*0.90)] # Données d'appentissage
Test = Data_data[int(len(Data_data)*0.90):] # Permet d'extraire par la suite une donnée de test
"""

Apprentissage = Data_data

# %% PRE-TEST DU DEVOIR KNN

l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,label = [],[],[],[],[],[],[],[],[],[],[]

f = open(r"C:\Users\thoma\Desktop\KNN\preTest.txt", "r") 
data = f.read()
lines = data.splitlines()
f.close()

l=[]
for i in range(len(lines)):
    l = lines[i].split(';')
    l1.append(l[0])
    l2.append(l[1])
    l3.append(l[2])
    l4.append(l[3])
    l5.append(l[4])
    l6.append(l[5])
    l7.append(l[6])
    l8.append(l[7])
    l9.append(l[8])
    l10.append(l[9])
    label.append(l[10])
    
Data_PreTest = []
for k in range(len(l1)):
    Data_PreTest.append([l1[k], l2[k], l3[k], l4[k], l5[k], l6[k], l7[k], l8[k], l9[k], l10[k], label[k]])

PreTest = Data_PreTest


# %% FINAL-TEST DU DEVOIR KNN

l1,l2,l3,l4,l5,l6,l7,l8,l9,l10 = [],[],[],[],[],[],[],[],[],[]

f = open(r"C:\Users\thoma\Desktop\KNN\finalTest.txt", "r") 
data = f.read()
lines = data.splitlines()
f.close()

l=[]
for i in range(len(lines)):
    l = lines[i].split(';')
    l1.append(l[0])
    l2.append(l[1])
    l3.append(l[2])
    l4.append(l[3])
    l5.append(l[4])
    l6.append(l[5])
    l7.append(l[6])
    l8.append(l[7])
    l9.append(l[8])
    l10.append(l[9])
    
Data_FinalTest = []
for k in range(len(l1)):
    Data_FinalTest.append([l1[k], l2[k], l3[k], l4[k], l5[k], l6[k], l7[k], l8[k], l9[k], l10[k]])

FinalTest = Data_FinalTest

# %% Code

# %% Distance euclidienne

def DistanceEuclidienne(x1, x2): # Implémentation de la mesure de la distance euclidienne
    distances = 0
    for i in range(len(x1)-1):
        distances += m.pow((float(x1[i]) - float(x2[i])), 2) 
    return np.sqrt(distances)

# %% Distance Manhattan 

def DistanceManhattan(x1, x2): # Implémentation de la mesure de la distance de Manhattan
    distances = 0
    for i in range(len(x1)-1):
        distances += abs(float(x1[i]) - float(x2[i]))
    return np.sqrt(distances)


# %% Voisins proches

# Code qui permet d'obtenir les k plus proches voisins

def Voisins(data, test, k, fonctionDist):
    """

    Parameters
    ----------
    data : list
        Contient les échantillons que nous voulons tester.
    test : tab
        Un des échantillons se trouvant dans les Test.
    k : int
        Correspond aux nombres des plus proches voisins que nous souhaitons obtenir.

    Returns
    -------
    voisins : list
        Contient les k plus proches voisins de l'échantillon testé.

    """
    distances = [] # Liste de stockage 
    
    # Grâce à la distance choisies, on calcule la distance entre l'échantillon test et les valeurs d'apprentissages
    for data_ligne in data:
        dist = fonctionDist(data_ligne[0:10], test[0:10]) # Pour les différents test réalisés sans le label à la fin
        distances.append((data_ligne, dist)) # On cree un tuple de la ligne des données d'apprentissage avec la distance qui les sépare de l'échantillon
        
    # On trie en fonction de la distance pour obtenir les K plus proches voisins
    distances.sort(key = lambda x: x[1])
    
    # Permet d'obtenir les K plus proches voisins, et donc leurs classes
    voisins = [] # Liste de stockage 
    for i in range(k): 
        voisins.append(distances[i][0]) # On ajoute la ligne de data qui est la i-ème plus proche de l'échantillon test
        
    return voisins

# %% KNN

# Application du KNN
# Permet de prédire la classe d'un échantillon test, donc cela correspond à la classe qui apparait le plus dans les voisins

def knn(Apprentissage, LigneDeTest, k, fonctionDist):
    """

    Parameters
    ----------
    Apprentissage : Liste
        Contient les données d'apprentissage
    LigneDeTest : List
        Correspond à l'échantillon de Test
    k : int
        k = Nombres de voisins choisis à sélectionnés
    fonctionDist : fonction
        Permet d'appliquer une fonction pour mesurer la distance

    Returns
    -------
    prediction : int
        Retourne la classe

    """
    voisins = Voisins(Apprentissage, LigneDeTest, k, fonctionDist) # On prend les k plus proches voinsins
    classes = [v[-1] for v in voisins] # On prend les classes de ces k plus proches voisins
    prediction = max(set(classes), key = classes.count) # On prédit la classe à partir de la calsse qui se répète le plus
    return prediction


# %% Déterminer la meilleur méthode de calcul de distance

def TestMeilleurFonctionDistance(Test, Apprentissage):
    kvoisins = []
    tauxEuclidienne = []
    tauxManhattan = []
    for k in range(1,100):
        kvoisins.append(k)
        tauxEuclidienne.append(0)
        tauxManhattan.append(0)
        for test in Test :
            if (knn(Apprentissage, test, k, DistanceEuclidienne) == test[10]) :
                tauxEuclidienne[k-1] += 1
            if (knn(Apprentissage, test, k, DistanceManhattan) == test[10]) :
                tauxManhattan[k-1] += 1
    if max(tauxManhattan) >= max(tauxEuclidienne) :
        print("La distance Manhattan est la meilleure.")
        return DistanceManhattan
    else : 
        print("La distance Euclidienne est la meilleure.")
        return DistanceEuclidienne
    return None

# Grâce aux premières data données : data et preTest, nous avons cherchés la méthode nous permettant d'obtenir la meilleure probabilité d'avoir un résultat correcte
# Cela nous a conduit à prendre la méthode Euclidienne pour calculer la distance

# %% Déterminer le meilleur K 

def TestDuMeilleureK(Test, Apprentissage, fonctionDist) :
    kvoisins = []
    taux = []
    for k in range(1,100):
        kvoisins.append(k)
        taux.append(0)
        for test in Test :
            if (knn(Apprentissage, test, k, fonctionDist) == test[10]) :
                taux[k-1] += 1
    return kvoisins, taux

def AffichageMeilleurK(k, taux):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(k, taux, 'o-')
    ymax = max(taux)
    xpos = taux.index(ymax)
    xmax = k[xpos]
    text = 'Maximun Local : ' + "k={:}, Max={:}".format(xmax, ymax)
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax, ymax-32), arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.title('Evolution de la quantité de label correcte en fonction de k', loc="left")
    plt.xlabel("Nombres de voisins = k")
    plt.ylabel("Nombres de bonnes correspondances")
    plt.grid(True)
    plt.show()

# Grâce aux premières data données : data et preTest, nous avons cherchés la méthode nous permettant d'obtenir le meilleure k avec la méthode Euclidienne précédement mis en avnat afin d'avoir un résultat correcte
# Nous affichons également cette évolution en fonction de k afin d'obtenir le meilleur k pour ces données
# Nous allons donc prendre k = 29, qui permet d'obtenir une probabilité de 0.718 de bon label
    
# %% Sauvegarde dans un fichier

# Permet de sauvegarder les classes obtenues dans le fichier .txt nommer marcal_groupeP

def SauvegarderDansFichier(Apprentissage, FinalTest, chemin):
    fichier = open(chemin, "w")
    k = 29
    fonctionDist = DistanceEuclidienne
    start = time.time()
    for testf in FinalTest :
        fichier.write(knn(Apprentissage, testf, k, fonctionDist) + "\n")
    end = time.time()
    fichier.close()
    print("\nFichier marcal_groupeP.txt créé")
    print("Temps d'ecxécution : ", str(round(end-start, 2)), "s")
    
# %% checkLabels de vérification

# Fonction fournit par le professeur afin de vérifier si notre fichier .txt créé respecte les consignes de 1000 lignes

def checkLabels(chemin):
    #code permettant de tester si un fichier de prédictions est au bon format.
    #il prend en paramètre un fichier de labels prédits
    #exemple d'utilisation > python checkLabels.py monFichierDePredictions.txt
    
    allLabels = ['0','1']
    #ce fichier s'attend à lire 1000 prédictions, une par ligne
    #réduisez nbLines en période de test.
    nbLines = 1000
    fd =open(chemin,'r')
    lines = fd.readlines()
    
    
    count=0
    for label in lines:
    	if label.strip() in allLabels:
    		count+=1
    	else:
    		break
    if count==nbLines:
    	print("Labels Check : Successfull!")
    else:
    	print("Wrong label line:"+str(count))
    	print("Labels Check : fail! ", nbLines, "predictions expected" ,count, "found")


# %%

if __name__ == '__main__':
    
    chemin = r'C:\Users\thoma\Desktop\KNN\marcal_groupeP.txt' # Chemin de sauvegarde du fichier
    
    ApprentissageComplet = Apprentissage + PreTest # On concatène ces deux ensembles de données afin d'avoir une meilleure base d'apprentissage et donc améliorer les résultats de notre KNN
    
    """
    # Permet d'obtenir la meilleur methode de calcul de distance
    TestMeilleurFonctionDistance(PreTest, Apprentissage)
    """
    
    """
    # Permet d'évaluer le meilleur k
    k, taux = TestDuMeilleureK(PreTest, Apprentissage, DistanceEuclidienne)
    AffichageMeilleurK(k, taux)
    """
    
    SauvegarderDansFichier(ApprentissageComplet, FinalTest, chemin) # On sauvegarde dans un fichier txt les labels obtenues
    checkLabels(chemin) # Permet d'illustrer que le fichier créer répond bien à ce que le professeur a demandé

