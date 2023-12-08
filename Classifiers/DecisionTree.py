import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline


class DecisionTreeWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.decision_tree = DecisionTreeClassifier()
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille
        self.param_grid = {
            'criterion': ['gini', 'entropy'],
    'max_depth': list(range(1, 7)),
    'min_samples_split': list(range(2, 8)),
    'min_samples_leaf': list(range(1, 8)),
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.decision_tree, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv):
        cv = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
        scores = cross_val_score(self.decision_tree, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv):
        cv = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])

        train_sizes, train_scores, test_scores = learning_curve(
            self.decision_tree, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Add shaded regions for the variability in training and test scores
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Decision Tree")
        plt.legend()
        plt.grid()
        train_accuracy = train_scores_mean[-1]
        print(f"Accuracy(train): {train_accuracy}")
        train.append(train_accuracy)

    def grid_search(self, X, y):
        best_cv = self.find_best_num_folds(X, y, cv_range=[3, 7])  # Modifier la plage de cv au besoin
        grid_search = GridSearchCV(self.decision_tree, self.param_grid, cv=best_cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.decision_tree = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

class DecisionTreeWithPCACrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.decision_tree = DecisionTreeClassifier()
        self.pca = PCA()
        self.accuracy = None  # Store accuracy

        # Define the grid of parameters for Decision Tree
        self.param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': list(range(1, 7)),
            'min_samples_split': list(range(2, 8)),
            'min_samples_leaf': list(range(1, 8)),
        }

        # Define the grid of parameters for PCA
        self.pca_param_grid = {
            'n_components': [100, 120, 130, 140, 160],
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.decision_tree, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv=None):
        if cv is None:
            cv=self.find_best_num_folds(X,y,cv_range=[3, 7])
        else:
            cv=cv
        # Combine PCA and Decision Tree into a pipeline
        pca_dt_pipeline = make_pipeline(self.pca, self.decision_tree)

        scores = cross_val_score(pca_dt_pipeline, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv):
        cv = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])

        train_sizes, train_scores, test_scores = learning_curve(
            self.decision_tree, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Plot mean training and test scores
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Add shaded regions for the variability in training and test scores
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Decision Tree")
        plt.legend()
        plt.grid()
        train_accuracy = train_scores_mean[-1]
        print(f"Accuracy(train): {train_accuracy}")
        train_pca.append(train_accuracy)


    def grid_search(self, X, y):
        best_cv = self.find_best_num_folds(X, y, cv_range=[3, 7])  # Modify the cv_range as needed

        # Grid search for PCA
        pca_grid_search = GridSearchCV(self.pca, self.pca_param_grid, cv=best_cv, n_jobs=self.n_jobs, scoring=self.scoring)
        pca_grid_search.fit(X, y)
        best_pca = pca_grid_search.best_estimator_

        # Apply PCA transformation
        X_pca = best_pca.transform(X)

        # Combine PCA and Decision Tree into a pipeline
        pca_dt_pipeline = make_pipeline(best_pca, self.decision_tree)

        # Grid search for Decision Tree
        dt_grid_search = GridSearchCV(self.decision_tree, self.param_grid, cv=best_cv, n_jobs=self.n_jobs, scoring=self.scoring)
        dt_grid_search.fit(X_pca, y)

        self.decision_tree = dt_grid_search.best_estimator_

        return pca_grid_search.best_params_, dt_grid_search.best_params_, dt_grid_search.best_score_


    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.clf = DecisionTreeClassifier()  # Utilisez le classifieur Decision Tree
        self.best_num_features = None
        self.best_accuracy = None
        self.best_cv = None  # Ajout pour stocker la meilleure valeur de cv

        # Définir la grille de paramètres pour la recherche en grille de Decision Tree
        self.clf_param_grid = {
           'criterion': ['gini', 'entropy'],
            'max_depth': list(range(1, 11)),
            'min_samples_split': list(range(1, 6)),
            'min_samples_leaf': list(range(2, 12, 2)),
        }

    def forward_selection(self, X, y, max_features, cv_range):
        best_accuracy = 0.0
        best_num_features = 1
        best_cv = None

        for num_features in range(1, max_features + 1):
            # Utiliser les 'num_features' premières colonnes comme caractéristiques
            X_selected = X[:, :num_features]

            # Déterminer le meilleur cv à utiliser pour cet ensemble de caractéristiques
            current_cv, _ = self.find_best_cv(X_selected, y, cv_range)

            # Ajuster le modèle Decision Tree
            self.clf.fit(X_selected, y)

            # Effectuer la validation croisée
            scores = cross_val_score(self.clf, X_selected, y, cv=current_cv, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_cv = current_cv

        # Définir le meilleur nombre de caractéristiques et le meilleur cv
        self.best_num_features = best_num_features
        self.best_cv = best_cv

        return X[:, :best_num_features]

    def find_best_cv(self, X, y, cv_range):
        best_cv = None
        best_test_accuracy = 0.0

        for cv in cv_range:
            scores = cross_val_score(self.clf, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
            test_accuracy = np.mean(scores)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_cv = cv

        return best_cv, best_test_accuracy

    def fit(self, X, y, cv_range):
        # Déterminer le meilleur nombre de caractéristiques et le meilleur cv à utiliser
        X_selected = self.forward_selection(X, y, X.shape[1], cv_range)
        
        # Ajuster le modèle Decision Tree avec le meilleur nombre de caractéristiques
        self.clf.fit(X_selected, y)

    def grid_search(self, X, y):
        # Recherche en grille des hyperparamètres Decision Tree
        clf_grid_search = GridSearchCV(self.clf, self.clf_param_grid, cv=self.best_cv, n_jobs=self.n_jobs, scoring=self.scoring)
        clf_grid_search.fit(X, y)
        best_clf = clf_grid_search.best_estimator_

        return best_clf, clf_grid_search.best_params_, clf_grid_search.best_score_

    def plot_learning_curve(self, X, y, train_sizes, cv_range):
        best_cv = None
        best_test_accuracy = 0.0

        for cv in cv_range:
            train_sizes, train_scores, test_scores = learning_curve(
                self.clf, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            current_test_accuracy = test_scores_mean[-1]  # Précision avec la dernière taille d'entraînement

            if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                best_cv = cv

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test")
        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("DecisionTree avec ForwardSelection")
        plt.legend()
        plt.grid()

        return best_cv

class DecisionTreeWithForwardSelectionAndGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.clf = DecisionTreeClassifier()  # Utilisez le classifieur Decision Tree
        self.best_num_features = None
        self.best_accuracy = None
        self.best_cv = None  # Ajout pour stocker la meilleure valeur de cv

        # Définir la grille de paramètres pour la recherche en grille de Decision Tree
        self.clf_param_grid = {
           'criterion': ['gini', 'entropy'],
            'max_depth': list(range(1, 11)),
            'min_samples_split': list(range(1, 6)),
            'min_samples_leaf': list(range(2, 12, 2)),
        }

    def forward_selection(self, X, y, max_features, cv_range):
        best_accuracy = 0.0
        best_num_features = 1
        best_cv = None

        for num_features in range(1, max_features + 1):
            # Utiliser les 'num_features' premières colonnes comme caractéristiques
            X_selected = X[:, :num_features]

            # Déterminer le meilleur cv à utiliser pour cet ensemble de caractéristiques
            current_cv = self.find_best_cv(X_selected, y, cv_range)

            # Ajuster le modèle Decision Tree
            self.clf.fit(X_selected, y)

            # Effectuer la validation croisée
            scores = cross_val_score(self.clf, X_selected, y, cv=current_cv, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_cv = current_cv

        # Définir le meilleur nombre de caractéristiques et le meilleur cv
        self.best_num_features = best_num_features
        self.best_cv = best_cv

        return X[:, :best_num_features]

    def find_best_cv(self, X, y, cv_range):
        best_cv = None
        best_test_accuracy = 0.0

        for cv in cv_range:
            scores = cross_val_score(self.clf, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
            test_accuracy = np.mean(scores)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_cv = cv

        return best_cv  

    def fit(self, X, y, cv):
        # Déterminer le meilleur nombre de caractéristiques et le meilleur cv à utiliser
        X_selected = self.forward_selection(X, y, X.shape[1], [cv])  # Wrap cv in a list
        # Ajuster le modèle Decision Tree avec le meilleur nombre de caractéristiques
        self.clf.fit(X_selected, y)

    def grid_search(self, X, y):
        # Recherche en grille des hyperparamètres Decision Tree
        clf_grid_search = GridSearchCV(self.clf, self.clf_param_grid, cv=self.best_cv, n_jobs=self.n_jobs, scoring=self.scoring)
        clf_grid_search.fit(X, y)
        best_clf = clf_grid_search.best_estimator_

        return best_clf, clf_grid_search.best_params_, clf_grid_search.best_score_

    def plot_learning_curve(self, X, y, train_sizes, cv):
        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Plot mean training and test scores
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Add shaded regions for the variability in training and test scores
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Decision Tree")
        plt.legend()
        plt.grid()
  
