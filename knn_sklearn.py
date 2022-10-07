import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
import matplotlib.pyplot as plt
import time

def affiche(x, a, n, k_voisins, nb_neighbors):
    # Création des deux tableaux (axe x et y) pour affichage des points de a (tableaux numpy)
    tabx_a = np.array([a[i][0] for i in range(n)])
    taby_a = np.array([a[i][1] for i in range(n)])

    # Création des deux tableaux (axe x et y) pour affichage des points de k_voisins (tableaux numpy)
    tabx_kv = np.array([k_voisins[0][i][0] for i in range(nb_neighbors)])
    taby_kv = np.array([k_voisins[0][i][1] for i in range(nb_neighbors)])

    # Initialisation des propriétés de l'affichage
    plt.scatter(tabx_a, taby_a, color='gray', label='a')  # a
    plt.scatter(tabx_kv, taby_kv, color='b', label='k_voisins et x')  # k_voisins
    plt.scatter(x[0][0], x[0][1], color='b', marker='x')  # x
    plt.legend()
    plt.title('k plus proches voisins')

    # Affichage
    plt.show()

def main():
    start_time = time.time()
    iris = datasets.load_iris()
    iris_data = iris.data[:, :2]
    iris_data_len = len(iris_data)
    x = [[2.2, 0.65]]
    k_list = [1, 10, 50, 100]
    for nb_neighbors in k_list:
        knn = NearestNeighbors(n_neighbors=nb_neighbors)
        knn.fit(iris_data)
        distances, indices = knn.kneighbors(np.array(x), return_distance=True)
        proches_voisins = [iris_data[x] for x in indices]



        print("k = " + str(nb_neighbors) + ": --- %s seconds ---" % (time.time() - start_time))
        affiche(x, iris_data, iris_data_len, proches_voisins, nb_neighbors)



# Programme principal
def main2():
    start_time = time.time()
    # Création des données

    iris = datasets.load_iris()
    iris_data = iris.data[:, :2]
    iris_data_len = len(iris_data)
    n = 13
    a = [[1.5, 2.75], [0.5, 2.75], [1, 3], [1, 2.5], [0.5, 3.5], [1.5, 3.25], [2, 2.75], [3.5, 4], [4.2, 4.5], [5, 4],
         [3.5, 4.5], [4, 4.15], [4.5, 4.25]]
    ...
    x = [2.2, 0.65]
    k_list = [1, 10, 50, 100]

    for k in k_list:
        # recherche des k plus proches voisins de x dans a
        # k_voisins = cherche_k_voisins(k, x, a, n)
        k_voisins = cherche_k_voisins(k, x, iris_data, iris_data_len)

        # affichage des k plus proches voisins

        print("k = " + str(k) + ": --- %s seconds ---" % (time.time() - start_time))
        affiche(x, iris_data, iris_data_len, k_voisins, k)

