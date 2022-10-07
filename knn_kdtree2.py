from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import time
import heapq


def distance(a, b):
    """Distance euclidienne entre deux points (espace à deux dimensions)

    Un point est représenté par un couple de valeurs flottantes.

    """
    distance_euclidean = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return distance_euclidean


def affiche(x, a, n, k_voisins, nb_neighbors):
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


def gen_kd_tree(data: list, col_index=0):
    next_col_index = 1 if col_index == 0 else 0

    if len(data) > 1:

        data.sort(key=lambda z: z[col_index])
        # median_index = get_median_index(len(data))
        median_index = len(data) >> 1

        median_point = data[median_index]

        leaf_right = data[median_index + 1:]
        leaf_left = data[:median_index]

        return [
            median_point,
            gen_kd_tree(leaf_left, next_col_index),
            gen_kd_tree(leaf_right, next_col_index),
        ]
    elif len(data) == 1:
        return [data[0], None, None]


def find_knn(kd_tree, target_point, k, nearest_neighbors=None, z=0):
    next_z = 1 if z == 0 else 0

    # if there is nothing in nearest neighbors -> being at root
    is_root = not nearest_neighbors

    # create an empty list when at root
    if is_root:
        nearest_neighbors = []
    # if kd tree is not null
    if kd_tree:
        # calculate the distance euclidean et distance by x or y
        node = kd_tree[0]
        dist_e = distance(node, target_point)
        dist_xy = abs(node[z] - target_point[z])

        if len(nearest_neighbors) < k:
            # if not find enough neighbors -> keep adding point to list
            heapq.heappush(nearest_neighbors, (-dist_e, node))
        elif dist_e < -nearest_neighbors[0][0]:
            # if found enough points -> compare new point distance e with the point in list that have the biggest dist e
            heapq.heappushpop(nearest_neighbors, (-dist_e, node))

        leaf_togo = [1, 2]
        if dist_xy > -nearest_neighbors[0][0]:
            # if distance of x or y is bigger than the dist e from x to the node -> only go left leaf and not right leaf
            del leaf_togo[1]

        for i in leaf_togo:
            find_knn(kd_tree[i], target_point, k, nearest_neighbors, next_z)

    if is_root:
        nearest_neighbors = [[h[1], -h[0]] for h in nearest_neighbors]
        return nearest_neighbors


def main():
    a = [[1, 3], [1, 8], [2, 2], [2, 10], [3, 6], [4, 1], [5, 4], [6, 8], [7, 4], [7, 7], [8, 2], [8, 5], [9, 9]]
    x = [4, 8]
    k = 5

    kdtree = gen_kd_tree(a, 0)
    kdtree_knn = find_knn(kdtree, x, k)

    ldsnksjdfnk = 0

    affiche(x, a, len(a), kdtree_knn, k)

main()