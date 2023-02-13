# MNIST-KNN

Sur des jeux de données de python (iris, wine, forest) ou des données écrites au hasard. 

Cherche à constamment améliorer la méthode pour trouver les k plus proches voisins d'un point. 
- L1 : méthode naïve 
Elle ne fonctionne que sur des petits jeux de données car sinon met trop de temps. 
- L2 : méthode avec des kdd tree 
En prenant la médiane entre chaque groupe de données que nous générons à la suite, nous créons un arbre de décision. 
- L3 : utilisation de la librarie sklearn de python 