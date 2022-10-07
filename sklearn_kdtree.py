from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# how leaf  size affect kd tree
# the smaller leaf size will generate a deeper tree adding cost to the construction time and to the tree traversal time
# https://stackoverflow.com/questions/63655219/kd-tree-meaning-of-leafsize-parameter
# https://stackoverflow.com/questions/65003877/understanding-leafsize-in-scipy-spatial-kdtree
# https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/


iris = datasets.load_iris()
iris_data = iris.data[:, [2, 3]].tolist()
iris_target = iris.target


# boston = datasets.load_boston()
# boston_data = boston.data[:, [5, 12]].tolist()
# boston_data_len = len(boston_data)

# diabetes = datasets.load_diabetes()
# diabetes_data = diabetes.data[:, [2, 8]].tolist()
# diabetes_data_len = len(diabetes_data)

# forest = datasets.fetch_covtype()
# forest_data = forest.data[:, [3, 4]].tolist()
# forest_target = forest.target
# forest_data_len = len(forest_data)

# 1: Spruce/Fir
# 2: Lodgepole Pine
# 3: Ponderosa Pine
# 4: Cottonwood/Willow
# 5: Aspen
# 6: Douglas-fir
# 7: Krummholz

# The amount of memory needed to store the tree scales as approximately n_samples / leaf_size
# For a specified leaf_size, a leaf node is guaranteed to satisfy leaf_size <= n_points <= 2 * leaf_size, except in the
# case that n_samples < leaf_size.

# class sklearn.neighbors.KDTree(X, leaf_size=40, metric='minkowski', **kwargs)¶

# X: n_samples is the number of points in the data set, and n_features is the dimension of the parameter space.
# Note: if X is a C-contiguous array of doubles then data will not be copied. Otherwise, an internal copy will be made.

# leaf_size

# metric: the distance metric to use for the tree

# Additional keywords are passed to the distance metric class.
#
def affiche(x, a, n, k_voisins, nb_neighbors):
    # Création des deux tableaux (axe x et y) pour affichage des points de a (tableaux numpy)
    tabx_a = np.array([a[i][0] for i in range(n)])
    taby_a = np.array([a[i][1] for i in range(n)])

    # Création des deux tableaux (axe x et y) pour affichage des points de k_voisins (tableaux numpy)
    tabx_kv = np.array([k_voisins[i][0] for i in range(nb_neighbors)])
    taby_kv = np.array([k_voisins[i][1] for i in range(nb_neighbors)])

    # Initialisation des propriétés de l'affichage
    plt.scatter(tabx_a, taby_a, color='gray', label='jeu de données')  # jeu de données
    plt.scatter(tabx_kv, taby_kv, color='b', label='k_voisins et x')  # k_voisins
    plt.scatter(x[0], x[1], color='b', marker='x')  # x
    plt.legend()
    plt.title('k plus proches voisins')

    # Affichage
    plt.show()


def sklearn_knn_kdtree_predict(x_train, x_test, y_train, y_test, nb_neighbors):
    knn_kdtree = KNeighborsClassifier(n_neighbors=nb_neighbors, algorithm='kd_tree', leaf_size=2, metric='euclidean')
    knn_kdtree.fit(x_train, y_train)

    return knn_kdtree


k = 20


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


import heapq


def find_knn(kd_tree, target_point, k, nearest_neighbors=None, z=0):
    next_z = 1 if z == 0 else 0

    # si nearest neighors vide -> is root TRUE
    is_root = not nearest_neighbors

    # si root TRUE -> créé une liste vide
    if is_root:
        nearest_neighbors = []
    # si kd_tree n'est pas nul
    if kd_tree:
        # calcul de la distance euclidienne et de la distance entre les x ou les y
        node = kd_tree[0]
        dist_e = distance(node, target_point)
        dist_xy = abs(node[z] - target_point[z])

        if len(nearest_neighbors) < k:
            # si nearest_neighbors toujours pas rempli
            # -> continuer de rajouter les points dans la liste
            heapq.heappush(nearest_neighbors, (-dist_e, node))
        elif dist_e < -nearest_neighbors[0][0]:
            # si nearest_neighbors rempli
            # -> comparer distance euclidienne des nouveaux points avec le point dans la liste qui a la plus grande distance euclidienne
            heapq.heappushpop(nearest_neighbors, (-dist_e, node))

        leaf_togo = [1, 2]
        if dist_xy > -nearest_neighbors[0][0]:
            # si distance entre les x ou les y est plus grande que la distance euclidiennede X et node
            # -> aller uniquement à gauche sinon aller à gauche et droite
            del leaf_togo[1]

        # effectuer cette fonction sur les feuilles qui suivent
        for i in leaf_togo:
            find_knn(kd_tree[i], target_point, k, nearest_neighbors, next_z)

    # une fois fini -> remettre dans l'ordre
    if is_root:
        nearest_neighbors = [[h[1], -h[0]] for h in nearest_neighbors]
        return nearest_neighbors


from math import sqrt


def distance(a, b):
    distance_euclidean = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return distance_euclidean


from collections import Counter


def projet_knn_kdtree(x_train, x_test, y_train, y_test, nb_neighbors):
    y_predict = []
    kd_tree = gen_kd_tree(x_train.copy())
    for target in x_test:
        knn = find_knn(kd_tree, target, k=nb_neighbors)
        result = Counter([y_train[x_train.index(i[0])] for i in knn]).most_common(1)[0][0]
        y_predict.append(result)
    return y_predict


def calculate_accuracy(y_pred, y_test):
    score_knn = 0

    for i, value in enumerate(y_test):
        if y_pred[i] == value:
            score_knn += 1

    return score_knn / len(y_test)


x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=0,
                                                    stratify=iris_target)

sklearn_result = sklearn_knn_kdtree_predict(x_train, x_test, y_train, y_test, nb_neighbors=k)
score_sklearn = sklearn_result.score(x_test, y_test)
print(score_sklearn)

projet_result = projet_knn_kdtree(x_train, x_test, y_train, y_test, nb_neighbors=k)
score_project = calculate_accuracy(projet_result, y_test)
print(score_sklearn)

import pandas as pd

df = pd.DataFrame({'lst1Title': score_sklearn,
                   'lst2Title': score_project,
                   'lst2Title': score_project,
                   'lst2Title': score_project,
                   'lst3Title': projet_result})
