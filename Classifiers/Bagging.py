import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif


class BaggingWithCrossValidationGridSearch:
    def __init__(self, base_estimator, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [3, 5, 7]  # Plage de CV par défaut
        self.clf = BaggingClassifier(base_estimator=base_estimator)
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille
        self.param_grid = {
        "base_estimator": [DecisionTreeClassifier()],
        "n_estimators": [5, 7, 10],
        "max_samples": [0.1, 0.25],
        "max_features": [0.2, 0.5],
         }

    def find_best_num_folds(self, X, y, cv_range=None):
        cv_values = cv_range if cv_range is not None else self.cv_range
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_values:
            scores = cross_val_score(self.clf, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv=None, class_weights=None):
            if cv is None:
                best_num_folds = self.find_best_num_folds(X, y, cv_range=self.cv_range)
                cv = best_num_folds

            scores = cross_val_score(self.clf, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
            self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Tracer les résultats moyens des entraînements et des tests
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats d'entrainement et de test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Bagging")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        grid_search = GridSearchCV(self.clf, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.clf = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

class PCAandBaggingWithCrossValidationGridSearch:
    def __init__(self, base_estimator, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [3, 5, 7]
        self.pca = PCA()
        self.clf = BaggingClassifier(base_estimator=base_estimator)
        self.accuracy = None

        # Updated PCA component grid
        self.pca_param_grid = {
            'n_components': [10, 20, 30, 40],
        }

        # Bagging hyperparameter grid
        self.clf_param_grid = {
            "base_estimator": [DecisionTreeClassifier()],
            "n_estimators": [5, 7, 10],
            "max_samples": [0.1, 0.25],
            "max_features": [0.2, 0.5],
        }

    def find_best_num_folds(self, X, y, cv_range=None):
        cv_values = cv_range if cv_range is not None else self.cv_range
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_values:
            scores = cross_val_score(self.clf, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds
        scores = cross_val_score(self.clf, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Tracer les résultats moyens des entraînements et des tests
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats d'entrainement et de test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Bagging")
        plt.legend()
        plt.grid()


    def grid_search(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        # Recherche en grille pour PCA
        pca_grid_search = GridSearchCV(self.pca, self.pca_param_grid, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        pca_grid_search.fit(X, y)
        best_pca = pca_grid_search.best_estimator_

        # Appliquer la transformation PCA
        X_pca = best_pca.transform(X)

        # Recherche en grille pour Bagging
        clf_grid_search = GridSearchCV(self.clf, self.clf_param_grid, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        clf_grid_search.fit(X_pca, y)
        best_clf = clf_grid_search.best_estimator_

        return best_pca, best_clf, pca_grid_search.best_params_, pca_grid_search.best_score_, clf_grid_search.best_params_, clf_grid_search.best_score_

class BaggingWithForwardSelectionAndGridSearch:
    def __init__(self, base_estimator, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [3, 5, 7]  # Plage de CV par défaut
        self.clf = BaggingClassifier(base_estimator=base_estimator)
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille de Bagging
        self.clf_param_grid = {
             "base_estimator": [DecisionTreeClassifier()],
            "n_estimators": [5, 7, 10],
            "max_samples": [0.1, 0.25],
            "max_features": [0.2, 0.5],
        }

    def forward_selection(self, X, y, max_features, cv_range=None):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]

        cv_values = cv_range if cv_range is not None else self.cv_range
        best_num_folds = self.find_best_num_folds(X, y, cv_range=cv_values)

        for num_features in range(1, max_features + 1):
            # Utiliser les 'num_features' premières colonnes comme caractéristiques
            X_selected = X[:, :num_features]

            # Ajuster le modèle Bagging
            self.clf.fit(X_selected, y)

            # Effectuer la validation croisée avec le nombre optimal de folds
            scores = cross_val_score(self.clf, X_selected, y, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected

        return best_X_selected, best_num_features, best_accuracy

    def find_best_num_folds(self, X, y, cv_range=None):
        cv_values = cv_range if cv_range is not None else self.cv_range
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_values:
            scores = cross_val_score(self.clf, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=self.cv_range)
            cv = best_num_folds

        scores = cross_val_score(self.clf, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.best_accuracy

    def get_best_num_features(self):
        return self.best_num_features

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

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
        plt.title("Bagging")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y):
        best_num_folds = self.find_best_num_folds(X, y, cv_range=self.cv_range)
        cv = best_num_folds
        # Recherche en grille pour Bagging
        clf_grid_search = GridSearchCV(self.clf, self.clf_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        clf_grid_search.fit(X, y)
        best_clf = clf_grid_search.best_estimator_

        return best_clf, clf_grid_search.best_params_, clf_grid_search.best_score_

