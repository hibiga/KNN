## k-voisins.py
## Introduction à Python
## Squelette du code recherche des k plus proches voisins
## @author: Eric Gouardères
## @date: novembre 2019

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from sklearn import datasets
import time


def distance(a, b):
    """Distance euclidienne entre deux points (espace à deux dimensions)

    Un point est représenté par un couple de valeurs flottantes.

    """
    distance_euclidean = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return distance_euclidean


def affiche(x, a, n, k_voisins, nb_neighbors):
    """Affichage en deux dimensions : d'un point x, et d'un nuage de points à partir d'une liste a de n points.

    Un point est représenté par un couple de valeurs flottantes.
    x et ses voisins sont en bleu, les autres points sont en gris.
    Exemple d'utilisation de la librairie matplotlib.

    """

    # Création des deux tableaux (axe x et y) pour affichage des points de a (tableaux numpy)
    tabx_a = np.array([a[i][0] for i in range(n)])
    taby_a = np.array([a[i][1] for i in range(n)])

    # Création des deux tableaux (axe x et y) pour affichage des points de k_voisins (tableaux numpy)
    tabx_kv = np.array([k_voisins[i][0][0] for i in range(nb_neighbors)])
    taby_kv = np.array([k_voisins[i][0][1] for i in range(nb_neighbors)])

    # Initialisation des propriétés de l'affichage
    plt.scatter(tabx_a, taby_a, color='gray', label='a')  # a
    plt.scatter(tabx_kv, taby_kv, color='b', label='k_voisins et x')  # k_voisins
    plt.scatter(x[0], x[1], color='b', marker='x')  # x
    plt.legend()
    plt.title('k plus proches voisins')

    # Affichage
    plt.show()


def cherche_k_voisins(k, x, a, n):
    """Recherche des k plus proches voisins d'un point x, dans une collection a de n éléments.

    a : liste.
    n et k : entiers tels que : 1 <= k < n.
    On considère que chaque élément représente un point d'un espace euclidien à deux dimensions.
    Un point est représenté par un couple de valeurs flottantes.

    """
    proches_voisins = []

    for i in range(k):
        dist_pv = distance(x, a[i])
        proches_voisins.append((a[i], dist_pv))

    proches_voisins.sort(key=lambda z: z[1], reverse=True)

    for i in range(k, n):
        dist = distance(x, a[i])
        if dist < proches_voisins[0][1]:
            del proches_voisins[0]
            proches_voisins.append((a[i], dist))
            proches_voisins.sort(key=lambda z: z[1], reverse=True)

    ...

    # proches_voisins contient les k plus proches voisins de x, les résultat est retourné
    return proches_voisins


# Programme principal
def main():

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

