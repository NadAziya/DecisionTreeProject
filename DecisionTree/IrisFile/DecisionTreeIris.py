# Importer les bibliothèques nécessaires
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Charger les données
data_iris = pd.read_csv('iris.csv')  # Ajuste le chemin si nécessaire
print(data_iris.head())  # Aperçu des premières lignes

# Préparer les données pour l'arbre de décision
# Séparer les caractéristiques (X) et la cible (y)
X = data_iris.drop(columns=['Species'])  # Toutes les colonnes sauf 'species'
y = data_iris['Species']  # Colonne cible

# Encodage des labels (conversion des noms des espèces en valeurs numériques)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convertit les noms d'espèces en 0, 1, 2

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner l'arbre de décision
clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # max_depth limite la profondeur
clf.fit(X_train, y_train)

# Afficher l'arbre de décision
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True
)
plt.title("Arbre de Décision - Dataset Iris")
plt.savefig('irisDecisionTree.png')  # Enregistrer l'arbre dans un fichier
plt.show()
