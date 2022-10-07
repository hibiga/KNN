## k-voisins.py
## Introduction à Python
## Squelette du code recherche des k plus proches voisins
## @author: Eric Gouardères
## @date: novembre 2019

import matplotlib.pyplot as plt
from math import sqrt
import numpy as np


def distance(a, b):
    """Distance euclidienne entre deux points (espace à deux dimensions)

    Un point est représenté par un couple de valeurs flottantes.

    """

    distance_euclidean = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return distance_euclidean

def cherche_k_voisins(k, x, a, n):
    proches_voisins = []

    for i in range(k):
        print(i)
        proches_voisins.append(a[i])

    for i in range(k, n):
        print(i)
        for j in range(k):
            if distance(a[i], x) < distance(proches_voisins[j], x):
                distance_max = distance(proches_voisins[0], x)
                indice_max = 0
                for l in range(k):
                    if distance_max < distance(proches_voisins[l], x):
                        distance_max = distance(proches_voisins[l], x)
                        indice_max = l
                del proches_voisins[indice_max]
                proches_voisins.append(a[i])
                break
    return proches_voisins


def gen_kd_tree(data: list, col_index=0):
    next_col_index = 1 if col_index == 0 else 0
    if len(data) > 1:
        data.sort(key=lambda z: z[col_index])
        median_index = len(data) >> 1

        median_point = data[median_index]

        # split array at median point
        leaf_right = data[median_index + 1:]
        leaf_left = data[:median_index]

        return [
            median_point,
            gen_kd_tree(leaf_left, next_col_index),
            gen_kd_tree(leaf_right, next_col_index),
        ]
    elif len(data) == 1:
        return [data[0], None, None]

# Programme principal

# Création des données
n = 13
a = [[1.5, 2.75], [0.5, 2.75], [1, 3], [1, 2.5], [0.5, 3.5], [2.5, 3.35], [2, 2.75], [3.5, 4], [4.2, 4.5], [5, 4],
     [3.5, 4.5], [4, 4.15], [4.5, 4.25]]
...
x = [2.55, 3.35]
k = 5

# recherche des k plus proches voisins de x dans a
k_voisins = cherche_k_voisins(k, x, a, len(a))

kdtree = gen_kd_tree(a)

b = 0



