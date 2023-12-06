from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


class AdaBoostWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, learning_rate=0.1)
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille
        self.param_grid = {
            'base_estimator__max_depth': [1, 2],  # Adjust as needed
            'base_estimator__min_samples_split': [1, 2],  # Adjust as needed
            'n_estimators': [50, 70],
            'learning_rate': [0.1, 0.5],  # Adjust as needed
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.ada_boost, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        scores = cross_val_score(self.ada_boost, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, cv_range)  # Use the best number of folds if num_folds is not provided
        train_sizes, train_scores, test_scores = learning_curve(
            self.ada_boost, X, y, train_sizes=train_sizes, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("AdaBoost")
        plt.legend()
        plt.grid()


    def grid_search(self, X, y, cv=None):  # Permet de spécifier le nombre de plis
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        grid_search = GridSearchCV(self.ada_boost, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.ada_boost = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

    def predict(self, X):
        return self.ada_boost.predict(X)

class PCAAdaBoostWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        
        # Add PCA to the pipeline
        self.ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        self.pca = PCA()
        self.pipeline = make_pipeline(self.pca, self.ada_boost)
        
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille
        self.param_grid = {
            'pca__n_components': [10, 20, 30, 40, 60],
            'adaboostclassifier__base_estimator__max_depth': [1, 2],
            'adaboostclassifier__n_estimators': [50, 70],
            'adaboostclassifier__learning_rate': [0.1, 1.0],
        }
    
    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0
 
        for num_folds in cv_range:
            scores = cross_val_score(self.pipeline, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)
 
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds
 
        return best_num_folds
 
    def fit(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds
            
        scores = cross_val_score(self.pipeline, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)
 
    def get_accuracy(self):
        return self.accuracy
 
    def plot_learning_curve(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            self.pipeline, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring,
            train_sizes=np.linspace(0.1, 1.0, 10), return_times=True
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
        plt.title("AdaBoost")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, cv=None):  # Permet de spécifier le nombre de plis
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds
            
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.ada_boost = grid_search.best_estimator_.named_steps['adaboostclassifier']
        self.pca = grid_search.best_estimator_.named_steps['pca']
        return grid_search.best_params_, grid_search.best_score_
 
    def predict(self, X):
        return self.ada_boost.predict(X)

class AdaBoostWithForwardSelectionAndGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille de AdaBoost
        self.ada_param_grid = {
            'base_estimator__max_depth': [1, 2],
            'n_estimators': [50, 70, 100],
            'base_estimator__min_samples_split': [2, 5],
            'learning_rate': [0.01, 0.1, 0.5],
        }

    def forward_selection(self, X, y, max_features):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]

        for num_features in range(1, max_features + 1):
            # Utiliser les 'num_features' premières colonnes comme caractéristiques
            X_selected = X[:, :num_features]

            # Ajuster le modèle AdaBoost
            self.ada_boost.fit(X_selected, y)

            # Effectuer la validation croisée
            scores = cross_val_score(self.ada_boost, X_selected, y, cv=5, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected

        return best_X_selected, best_num_features, best_accuracy

    def fit(self, X, y):
        # Déterminer le meilleur nombre de caractéristiques à utiliser
        X_selected, best_num_features, best_accuracy = self.forward_selection(X, y, X.shape[1])

        # Ajuster le modèle AdaBoost avec le meilleur nombre de caractéristiques
        self.ada_boost.fit(X_selected, y)
        self.best_num_features = best_num_features
        self.best_accuracy = best_accuracy

    def get_accuracy(self):
        return self.best_accuracy

    def get_best_num_features(self):
        return self.best_num_features
    
    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            # Perform cross-validation without stratification
            scores = cross_val_score(self.ada_boost, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds
    
    def plot_learning_curve(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.ada_boost, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("AdaBoost")
        plt.legend()
        plt.grid()


    def grid_search(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        # Create a base estimator (you can choose a different one based on your data)
        base_estimator = DecisionTreeClassifier()

        # Define the parameter grid including the base estimator
        param_grid = {
            'base_estimator': [base_estimator],
            'n_estimators': self.ada_param_grid['n_estimators'],
            'learning_rate': self.ada_param_grid['learning_rate'],
        }

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(self.ada_boost, param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)

        # Get the best estimator and its parameters
        best_ada = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_ada, best_params, best_score

    def predict(self, X):
        return self.ada_boost.predict(X)

