from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


def distance(a, b):
    distance_euclidean = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return distance_euclidean


def get_kd_tree(points, leaf_list=[]):
    if len(points) > 1:
        points = sorted(points, key=lambda k: [k[0], k[1]])
        half = len(points) >> 1
        leaf_list.append([points[half], half])

        return [
            get_kd_tree(points[: half], leaf_list),
            get_kd_tree(points[half + 1:], leaf_list),
            points[half],
            leaf_list
        ]
    elif len(points) == 1:
        return [None, None, points[0], leaf_list]



# k nearest neighbors
def get_knn_kd(points, x, tree, k):
    voisins = []

    distance_list = []
    for c in tree[3]:
        dist_e1 = distance(x, c[0])

        distance_list.append(([c[0], dist_e1], c[1]))
    distance_list.sort(key=lambda z: z[1])
    mini = distance_list[0][0][0]
    point_index = distance_list[0][1]

    points = sorted(points, key=lambda k: [k[0], k[1]])
    #point_index = points.index(mini)

    i = j = 1
    voisins.append(mini)
    k = k - 1
    while k != 0:
        if (point_index - i) < 0:
            voisins.append(points[point_index + j])
            j = j + 1
        # elif points[point_index + j] is None:
        elif (point_index + j) >= len(points):
            voisins.append(points[point_index - i])
            i = i + 1
        else:
            d1 = distance(x, points[point_index - i])
            d2 = distance(x, points[point_index + j])
            if d1 >= d2:
                voisins.append(points[point_index + j])
                j = j + 1
            else:
                voisins.append(points[point_index - i])
                i = i + 1
        k = k - 1

    return voisins


def affiche(x, a, n, k_voisins, nb_neighbors):
    # Création des deux tableaux (axe x et y) pour affichage des points de a (tableaux numpy)
    tabx_a = np.array([a[i][0] for i in range(n)])
    taby_a = np.array([a[i][1] for i in range(n)])

    # Création des deux tableaux (axe x et y) pour affichage des points de k_voisins (tableaux numpy)
    tabx_kv = np.array([k_voisins[i][0] for i in range(nb_neighbors)])
    taby_kv = np.array([k_voisins[i][1] for i in range(nb_neighbors)])

    # Initialisation des propriétés de l'affichage
    plt.scatter(tabx_a, taby_a, color='gray', label='a')  # a
    plt.scatter(tabx_kv, taby_kv, color='b', label='k_voisins et x')  # k_voisins
    plt.scatter(x[0], x[1], color='b', marker='x')  # x
    plt.legend()
    plt.title('k plus proches voisins')

    # Affichage
    plt.show()