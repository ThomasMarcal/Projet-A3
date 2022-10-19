"""

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Projet IA - STAR WARS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MARCAL Thomas
Groupe de TD : P

"""

# %% Bibliothèques utilisées

import random
import pandas
import numpy as np
import matplotlib.pyplot as plt
import time


# %% Importation des données

# On extraire les données de chaque colones du csv
colnames = ['t', 'x', 'y']

# Utilisation de la bibliothèque Pandas
data = pandas.read_csv(
    r'C:\Users\thoma\Desktop\ESILV\INFORMATIQUE\Datascience & IA\IA STAR WARS - FINAL\position_sample.csv', sep=";", names=colnames)

x = data.x.tolist()
y = data.y.tolist()
t = data.t.tolist()

# On enlève le nom des colonnes
x.pop(0)
y.pop(0)
t.pop(0)

# On transforme la liste de str en float
X = list(map(float, x))
Y = list(map(float, y))
T = list(map(float, t))


# %% Données

def fx(t, p1, p2, p3): # orbite de Lissajous pour x(t)
    return p1*np.sin(p2*t+p3)

def fy(t, p4, p5, p6): # orbite de Lissajous pour y(t)
    return p4*np.sin(p5*t+p6)


nbInconnues = 6 # Nombre d'inconnues de notre problème qui correspond au p_i

eps = 0.04 # Précision souhaité pour la fitness


# %% Création d'une classe individu

class individu:
    
    def __init__(self, val = None): # Permet d'initialiser notre individu
        if val==None:
            self.val = [random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)]
        else:
            self.val = val
    
    def fitness(self, T, X, Y):   # Permet d'initialiser notre fonction de fitness
        fit = []                  # Création d'une liste de stockage
        for i in range(len(T)):   # Permet de calsuler la fonction de fitness pour chaque ligne du CSV 
            fit.append((X[i] - self.val[0] * np.sin(self.val[1] * T[i] + self.val[2]))**2 + (Y[i] - self.val[3] * np.sin(self.val[4] * T[i] + self.val[5]))**2) # Utilisation de la norme euclidienne au carré
        return sum(fit)/len(fit)  # On fait la moyenne de notre liste, pour obtenir la fitness  


# %% Code

def population(n): 
    """
    Création de la population contenant n individu(s)
    """
    pop = []                   # Création de la liste pop qui va stocker les n individu(s)
    for i in range(n):
        pop.append(individu()) # On ajoute un individu à la liste
    return pop                 # On renvoit la liste pop




def evaluate(pop): 
    """
    On évalue notre population en fonction du cout de chaque individu
    """
    return sorted(pop, key = lambda x: x.fitness(T, X, Y)) # On trie notre population en fonction de la fitness de chaque individu




def selection(pop, hcount, lcount): # La sélection a lieu après l'évaluation
    """
    On prend notre population évalué, puis nous renvoyons les hcount premier 
    élément de la liste évalué et qui correspondent au mieux à la fitness et 
    les lcount derniers éléments, qui respectent le moins bien la fonction de 
    fitness
    """
    if (len(pop) <= hcount + lcount): # On vérifie si les dimensions correspondent
        return pop
    else:
        return [pop[i] for i in range(hcount)] + [pop[i] for i in range(len(pop)-lcount,len(pop))]
  
        
  
    
def croisement(i1,i2): 
    """
    Retourne une liste de deux individus à partir de deux individus i1 et i2 
    (3 premières données de i1 suivies des 3 dernières de i2, puis les 3 
     premières données de i2 suivies des 3 dernières de i1)
    """
    j1 = individu()    # Création d'individus de stockage
    j2 = individu()
    for i in range(nbInconnues):
        if i < nbInconnues//2:
            j2.val[i]=i2.val[i]
            j1.val[i]=i1.val[i]
        else:
            j1.val[i]=i2.val[i]
            j2.val[i]=i1.val[i]
    return [j1,j2] # Retourne les 2 individus nouvellement créés à partir de j1 et j2




def mutation(ind, lim): 
    """
    On modifie un élément de ind aléatoire, auxquelle on injecte une nouvelle valeur
    """
    index = random.randint(0, nbInconnues-1) # On prend un indice random 
    if (lim > 60) :
        val = random.uniform(-100, 100)
    else :
        val = random.uniform(ind.val[index]-0.1, ind.val[index]+0.1)  
    ind.val[index] = val 
    return ind # On retourne l'individu
    


# %% Solution

def AlgoGen(T,X,Y,eps):
    start = time.perf_counter()  # Permet de mesurer le temps
    cpt = 0                      # Permet de compter le nombre d'itérations nécéssaires
    pop = population(150)        # On crée une population aléatoire d'individu(s) 
    
    while True:                  # On entre dans une boucle infinie tant que aucune solution n'est trouvé  
        cpt += 1                 # On incrément le compteur
        evaluation=evaluate(pop)               # On évalue notre population, on la trie en fonction du fitness
        # print(evaluation[0].fitness(T,X,Y))
        
        if evaluation[0].fitness(T,X,Y) > eps: # Si on ne trouve pas de solution      
            select=selection(evaluation,40,4)  # On sélectionne les 40 meilleurs et les 4 pires solutions
            croises=[]       
            for i in range(0,len(select),2):   # On croise notre population 2 par 2
                croises+=croisement(select[i],select[i+1])
            mutes=[]           
            for i in select:                   # On opère la mutation sur chaque individus sélectionnés
                mutes.append(mutation(i, evaluation[0].fitness(T,X,Y)))
            newalea = population(18)           # On ajoute 18 nouveaux individu(s) aléatoires à la population
            pop=select[:]+croises[:]+mutes[:]+newalea[:] 
            
        else:                         # Si on trouve une solution            
            end = time.perf_counter() # Mesure le temps de fin de l'algorithme
            delay = end - start       # On obtient la durée 
            print("\nDurée : ", delay)
            print("\nNombre d'itérations : ", cpt)
            return evaluation[0].val  # Renvoie la solution optimal qui espcete la contrainte epsilon
        
    return "Error"



# %% Solution au problème

solution = AlgoGen(T,X,Y,eps) # L'ensemble des solutions p_i du probleme

p1 = solution[0]
p2 = solution[1]
p3 = solution[2]%(2*np.pi) # modulo 2*pi car on a un sin et que sinon p3 n'appartient pas à [-100 ; 100]
p4 = solution[3]
p5 = solution[4]
p6 = solution[5]%(2*np.pi) # modulo 2*pi car on a un sin et que sinon p6 n'appartient pas à [-100 ; 100]    

# Affichage des solutions
print("\nCombinaison de paramètres expliquant au mieux la trajectoire du satellite : ")
print("p1 = ", p1, "\np2 = ", p2, "\np3 = ", p3, "\np4 = ", p4, "\np5 = ", p5, "\np6 = ", p6)
 

FX = []
FY = []
deltaX = []
deltaY = []
for i in range(len(T)):
    FX.append(fx(T[i], p1, p2, p3))
    FY.append(fy(T[i], p4, p5, p6))
    deltaX.append(X[i] - fx(T[i], p1, p2, p3))
    deltaY.append(Y[i] - fy(T[i], p4, p5, p6))

"""
print("\n==> Précision pour la valeur de X : ", abs(sum(deltaX)/len(deltaX)))
print("\n==> Précision pour la valeur de Y : ", abs(sum(deltaY)/len(deltaY)))

"""


# %% Affichage de la trajectoire et des coordonnées

t = np.linspace(0, 2*np.pi, 10000)

plt.scatter(X, Y, s = 150, c = 'g', marker = '+', label= 'Théoriques')
plt.scatter(FX, FY, s = 150, c = 'r', marker = '+', label= 'Calculés')
plt.plot(fx(t, p1, p2, p3),fy(t, p4, p5, p6), c = 'k', label = 'Trajectoire')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title("Projet IA - STAR WARS")

plt.show()

# %matplotlib auto


# %% Code permettant de répondre à la QUESTION 5 du rapport

"""

def Convergence(T,X,Y):
    cpt = 0                      # Permet de compter le nombre d'itérations nécéssaires
    pop = population(150)        # On crée une population aléatoire d'individu(s) 
    deltaX = []
    deltaY = []
    sommeX = 0
    sommeY = 0
    while True:                  # On entre dans une boucle infinie tant que aucune solution n'est trouvé  
        cpt += 1                 # On incrément le compteur
        evaluation=evaluate(pop)               # On évalue notre population, on la trie en fonction du fitness 
        
        # Permet de calculer la convergence de la valeur X et Y
        for i in range(len(X)):
            sommeX += abs(X[i]-evaluation[0].val[0]*np.sin(evaluation[0].val[1]*T[i]+evaluation[0].val[2]))**2 
            sommeY += abs(Y[i]-evaluation[0].val[3]*np.sin(evaluation[0].val[4]*T[i]+evaluation[0].val[5]))**2
        deltaX.append(sommeX/30)
        deltaY.append(sommeY/30)
        
        if evaluation[0].fitness(T,X,Y) > eps: # Si on ne trouve pas de solution      
            select=selection(evaluation,40,4)  # On sélectionne les 40 meilleurs et les 4 pires solutions
            croises=[]       
            for i in range(0,len(select),2):   # On croise notre population 2 par 2
                croises+=croisement(select[i],select[i+1])
            mutes=[]           
            for i in select:                   # On opère la mutation sur chaque individus sélectionnés
                mutes.append(mutation(i, evaluation[0].fitness(T,X,Y)))
            newalea = population(18)           # On ajoute 18 nouveaux individu(s) aléatoires à la population
            pop=select[:]+croises[:]+mutes[:]+newalea[:]       
        else:                         # Si on trouve une solution            
            return deltaX,deltaY  # Renvoie la solution optimal qui espcete la contrainte epsilon
    return "Error"


conv = Convergence(T,X,Y)

TX = conv[0]
TY = conv[1]
Iter = range(0, len(TX))

plt.plot(Iter, TX, c = 'r', label = "Convergence de X")
plt.plot(Iter, TY, c = 'g', label = "Convergence de Y")
plt.xlabel("Itérations")
plt.legend(loc = "lower right")
plt.title("Convergence des solutions en fonction du nombre d'itérations")
plt.show()

"""


