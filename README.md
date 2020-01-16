# PLP_project
L’objectif est d’étudier l’impact du calcul parallèle dans le cas ou nous avons des données massives ou volumineuses. L’étude vise principalement à comparer les performances deux bibliothèques d’apprentissage automatique : scikit-learn de python sans parallélisation et MLlib de pyspark pour le calcul parallèle.

# 1. Présentation des données:
Nous disposons d’un jeu de données représentant quelques propriétés pétrophysiques des sols : variables explicatives ou inputs, 
que nous allons analyser afin de déduire l’existence ou non de quatre métaux principaux : variable à prédire ou
inputs.
Le jeu de données comporte 11 colonnes dont 5 inputs (propriétés pétrophysqiues) et 6 outputs (pourcentage du minéral dans le 
sol) et 365275 individus (sols à explorer). Les outputs vont être transformés en une matrice creuse en
mettant 0 si le pourcentage du minéral est en dessous d’un certain seuil et 1 si le pourcentage est au dessus. Notre modèle 
d’apprentissage automatique devra dans les deux cas ( Sklearn de Python et MLlib de PySpark) prédire cette matrice
output de 0 et de 1. L’utilité de cette étude réside dans le fait de pouvoir classifier les sols à étudier en vue d’un 
potentielle exploration pétrolière qu’on ne traitera pas dans ce rapport.

# 2. Étapes de traitement des données:
# 2.1  Pré-traitement:

Comme discuté dans la présentation des données, il nous faut binariser les outputs, c’est à dire transformer les valeurs 
flottantes qui représentent la proportion du minéral dans l’échantillon du sol considéré en 0 ou 1. On se base sur un
seuil pré-indiqué pour chaque minéral qui est en relation directe avec la présence de pétrole et qui peut être trouvé dans la 
littérature pétrophysique. Pour les inputs il nous faut les normaliser, en d’autres termes les rendre
centrées (moyenne nulle) et les réduire (variance unité), cela ce fait rapidemment par la fonction StandardScaler de 
Sklearn.preprocessing ou pyspark.ml.feature.

# 2.2 Modèle et tuning des hyperparamétre:
La fonction de perte d’un modèle statistique peut être exprimée par la somme du biais quadratique de ses prédictions, de la 
variance de ces prédictions et de la variance des termes d’erreur $\epsilon$. Étant donné que le biais et la variance au
carré et les termes d’erreur sont positifs, ce qui rend compte du caractère aléatoire des données, est au delà de notre 
contrôle, nous réduisons L’erreur quadratique en minimisant la variance et le biais de notre modèle en choisissant le modèle de
classification Random Forest (Forêt aléatoire).
Le tuning des hyperparmétre (GridSearch et cross validation) pour maximiser le score (accuracy/précision) sur le l’échantillon 
du test nous a conduit vers des arbres de longueur égale à 20 avec un nombre d’estimateurs égale à 150.

# 3. Études des performances:
# 3.1 Suréchantillonnage:
notre problème de classification affiche un certain niveau de déséquilibre
de classe, c’est-à-dire que les classes ne constituent pas une partie égale de notre
ensemble de données. Il est important d’ajuster correctement nos mesures et méthodes afin de les rendre concordants avec nos 
objectifs.
Un moyen simple de corriger le déséquilibre de classes consiste simplement à les équilibrer, soit en sur-échantillonnant les instances de la classe minoritaire, soit en sous-échantillonnant les instances de la classe majoritaire. Cela nous permet
simplement de créer un ensemble de données équilibrées qui, en théorie, ne devrait
pas conduire à des classificateurs biaisés en faveur d’une classe ou d’une autre.
Cependant, dans la pratique, ces méthodes d’échantillonnage simples présentent
des défauts : Le suréchantillonnage de la minorité peut conduire à un surajustement du modèle, car il introduira des 
instances en double, puisant dans un pool d’instances déjà réduit. De même, le sous-échantillonnage de la majorité peut
finir par laisser de côté des instances importantes qui fournissent des différences
importantes entre les deux classes. Il existe également des méthodes d’échantillonnage plus puissantes qui vont
au-delà du simple sur-échantillonnage ou du sous-échantillonnage. L’exemple le plus connu est SMOTE, qui crée de nouvelles 
instances de la classe minoritaire en formant des combinaisons convexes d’instances voisines. Comme le montre le
graphique ci-dessous, il trace efficacement des lignes entre des points minoritaires dans l’espace des features et des 
échantillons le long de ces lignes. Cela nous permet d’équilibrer notre ensemble de données sans trop de sur adaptation, car
nous créons de nouveaux exemples synthétiques plutôt que d’utiliser des doublons. Cela n’empêche toutefois pas tout le risque 
de sur-ajustement, car ils sont toujours créés à partir de points de données existants.
# 3.2 Temps de calcul:
# 3.2.1 Pré-traitement:
Nous remarquons que l’accroissement du temps de calcul de l’étape de Prétraitement est logarithmique par rapport au volume de data à traiter. Les temps
de calculs en fonction du nombre de cœurs ou du nombre d’exécuteurs est en faveur
de l’architecture distribuée.
# 3.2.2 Apprentissage:
Le temps de calcul dans la phase d’apprentissage de Spark est énorme ce qui laisse penser que Spark n’est pas le meilleur 
choix pour la phase d’apprentissage sauf dans le cas où vous ne disposez pas de suffisamment de RAM pour allouer de
la mémoire aux données. Sklearn fournit de très hautes performances.
# 4. Conclusion:
A travers ce projet de comparaison de performances entre architecture distribuée type Map-Reduce (par exemple sur MLlib de 
PySpark) et l’architecture standard (Par exemple sur Sklearn sur Python), on comprend que l’une peut être
plus adaptée que l’autre suivant de la nature du jeu de données dont on dispose, sa taille, le type de données qu’il 
contient, les caractéristiques intrinsèques de la machine sur lequel le calcul est effectué (mémoire, processeurs) qu’elle 
soit virtuelle ou physique.
Dans notre cas précis (jeu de données pétrophysiques) la parallélisation avait fournis de bons résultats pour le pré-
traitement des données mais l’architecture standard l’a surpassé en apprentissage et cela vient du fait que les données ne 
sont pas aussi massifs qu’ils devraient l’être mais aussi de la non utilisation de machine virtuelle comportant plusieurs 
buffers/clusters qui pourrait décroitre nettement le temps de calcul.
