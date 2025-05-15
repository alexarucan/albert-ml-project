 ➡️ Vous pouvez accéder aux slides de présentations en cliquant sur ce lien : [Lien vers le lien Canva](https://www.canva.com/design/DAGmTqkTG04/YcxeVjaBN-3g7aRvpZ-ZVw/view?utm_content=DAGmTqkTG04&utm_campaign=share_your_design&utm_medium=link2&utm_source=shareyourdesignpanel)
 (ou simplement dans le dossier "presentation")


 ---
 
 ## Objectif du projet
 
 Les fichiers de données nécessaires pour entraîner et tester le modèle ne sont **pas inclus dans le dépôt Git** pour des raisons de taille et de confidentialité.
 Ce projet vise à prédire si une tumeur du sein est bénigne ou maligne à partir de données médicales extraites d’images (ponction à l’aiguille fine). Le problème est formulé comme une tâche de classification binaire supervisée.
 
 ---
 
 ## Data utilisées
 
 Le jeu de données utilisé provient du dataset *Breast Cancer Wisconsin (Diagnostic)*, disponible publiquement sur Kaggle. Les fichiers de données nécessaires pour entraîner et tester le modèle ne sont **pas inclus dans le dépôt Git** pour des raisons de taille et de confidentialité.
 ➡️ Vous pouvez les télécharger ici : [Lien vers le dossier Google Drive](https://drive.google.com/drive/folders/12JTnZtRQbY2GckIkGjPtzhvGKnOrSTf8?usp=share_link)
 
 Une fois les fichiers téléchargés, placez-les dans le dossier suivant :
 Chaque ligne correspond à une tumeur, et les 30 variables numériques représentent différentes caractéristiques des cellules (par exemple : rayon, texture, périmètre, symétrie, etc.). La variable cible est `diagnosis`, indiquant si la tumeur est bénigne (0) ou maligne (1).
 
 Nous avons utilisé ce dataset dans trois versions différentes :
 - Une version propre `Sans_bruit.ipynb`, sans bruit, pour établir une première baseline.
 - Une version modifiée avec des valeurs manquantes `Avec_bruit_NaN_drop.ipynb` insérées aléatoirement, puis supprimées via `dropna`.
 - Une dernière version avec les mêmes `NaN`,`Avec_bruit_NaN_imputation.ipynb` mais cette fois imputées automatiquement en fonction du type de variable (moyenne, médiane ou KNN).
 
 Cela nous a permis d’analyser non seulement les performances des modèles dans des conditions idéales, mais aussi leur robustesse face à des données incomplètes, ce qui est plus proche de la réalité en contexte médical.
 
 ---
 
 ## Méthodes de machine learning utilisées
 
 Dans notre projet, on a testé trois modèles de classification supervisée pour prédire si une tumeur est maligne ou bénigne. Le but était de comparer leurs performances, d’abord sur les données sans bruit, puis sur des versions du dataset modifiées avec du bruit.
 
 - **Régression logistique** :  
   C’est le modèle le plus simple, mais souvent très efficace. On l’a utilisé comme modèle de référence (baseline), car il est rapide à entraîner et permet de bien comprendre l’impact de chaque variable. Malgré sa simplicité, il a donné d’excellents résultats sur notre jeu de données propre.
 
 - **Random Forest** :  
   Ce modèle est composé de plusieurs arbres de décision, ce qui le rend très robuste, même si les données sont bruitées ou qu’il y a des outliers. Il permet aussi de savoir quelles variables sont les plus importantes pour la prédiction. On l’a choisi pour tester un modèle plus complexe et non linéaire. Il a bien résisté à l’introduction de NaN, notamment grâce à son fonctionnement en ensemble.
 
 - **SVM (Support Vector Machine)** :  
   Le SVM est un modèle qui marche bien quand les classes sont bien séparées, ce qui est le cas ici avec certaines variables très discriminantes. On a testé différents noyaux (linéaire et RBF). Il est un peu plus long à entraîner mais ses performances ont été très proches de celles des autres modèles.
 
 On a utilisé `GridSearchCV` avec validation croisée pour trouver les meilleurs hyperparamètres pour chaque modèle. Tous les modèles ont été évalués avec **accuracy, precision, recall et F1-score**. Magré de bons résultats il est important de bien équilibrer les faux positifs et les faux négatifs car nous sommes dans un contexte médical.
 
 
 ---


## Application Streamlit
Nous avons développé une application interactive en Streamlit (app.py) pour exploiter notre modèle de prédiction dans une interface simple et pédagogique. Elle propose deux onglets.

Le premier, Prédicteur de tumeur, permet de simuler un cas médical en ajustant les variables mesurées sur les noyaux cellulaires. Le modèle Random Forest retourne une probabilité de malignité, interprétée selon trois zones de risque (faible, modéré, élevé), accompagnée d’un radar chart pour visualiser le profil de la tumeur.

Le second, Robustesse des modèles, permet d’injecter du bruit (valeurs manquantes) dans les données pour tester la stabilité des prédictions. Trois modèles sont comparés après imputation (régression logistique, SVM, Random Forest) à l’aide d’indicateurs de performance et de matrices de confusion.

L’application met en lumière les limites et la fiabilité des prédictions dans des contextes réalistes, tout en restant accessible à un public non technique.


---

Ce projet nous a permis d’aborder un cas concret de classification binaire appliqué à un enjeu médical sensible : la détection précoce du cancer du sein. Nous avons comparé plusieurs modèles, testé leur robustesse à des données incomplètes et développé une application interactive pour rendre le modèle interprétable et accessible.

Au-delà des performances techniques, ce travail met en évidence l’importance de la transparence, de l’explicabilité et de la prudence dans les systèmes d’aide à la décision médicale. Des pistes d’amélioration restent ouvertes, notamment l’intégration de données issues d’autres sources, l’optimisation du déploiement de l’application ou encore l’ajout de techniques de détection d’incertitude.