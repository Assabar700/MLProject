import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

class RandomForestWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.rf = RandomForestClassifier()
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille
        self.param_grid = {
            'n_estimators': [50, 70],
            'max_depth': [None, 5, 10],
            'min_samples_split': [1, 2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.rf, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5])  
            cv = best_num_folds

        # Define the base RandomForestClassifier without specific hyperparameters
        base_rf = RandomForestClassifier()

        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(base_rf, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)

        # Use the best hyperparameters for the RandomForestClassifier
        self.rf = grid_search.best_estimator_

        # Calculate accuracy using the best hyperparameters
        scores = cross_val_score(self.rf, X, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)

    def get_accuracy(self):
        return self.accuracy
    
    def plot_learning_curve(self, X, y,train_sizes,num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, cv_range)  # Use the best number of folds if num_folds is not provided
        train_sizes, train_scores, test_scores = learning_curve(
            self.rf, X, y, train_sizes=train_sizes, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring
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
        # Adjust y-axis ticks to start from 0
        y_ticks = plt.yticks()[0]

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Random Forest")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):  # Permet de spécifier le nombre de plis
        if num_folds is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5])  
            num_folds = best_num_folds
            
        grid_search = GridSearchCV(self.rf, self.param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X, y)
        self.rf = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
    
    def predict(self, X):
        return self.rf.predict(X)

class PCAandRandomForestWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [3, 5]  # Default CV range
        self.pca = PCA()
        self.random_forest = RandomForestClassifier()
        self.accuracy = None  # Store accuracy
        self.best_num_folds = None  # Store the best number of folds
 
        # Define parameter grid for grid search
        self.pca_param_grid = {
            'n_components': [100, 120, 130, 140, 160],  # Customize the list according to your needs
        }
 
        self.random_forest_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
 
    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        self.best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return self.best_num_folds
 
    def fit(self, X, y, cv=1):
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        scores = cross_val_score(self.random_forest, X, y, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)
 
    def get_accuracy(self):
        return self.accuracy
    
    def plot_learning_curve(self, X, y,train_sizes,cv=None):
        if cv is None:
            cv = self.find_best_num_folds(X, y, cv_range)  # Use the best number of folds if num_folds is not provided
        train_sizes, train_scores, test_scores = learning_curve(
            self.random_forest, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("Random Forest")
        plt.legend()
        plt.grid()
 
    def grid_search(self, X, y, cv=None):
        if cv is None:
            cv = self.best_num_folds  # Use the stored best number of folds if not provided
        # Grid search for PCA
        pca_grid_search = GridSearchCV(self.pca, self.pca_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        pca_grid_search.fit(X, y)
        best_pca = pca_grid_search.best_estimator_
 
        # Apply PCA transformation
        X_pca = best_pca.transform(X)
 
        # Grid search for Random Forest
        rf_grid_search = GridSearchCV(self.random_forest, self.random_forest_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        rf_grid_search.fit(X_pca, y)
        best_rf = rf_grid_search.best_estimator_
 
        return best_pca, best_rf, pca_grid_search.best_params_, pca_grid_search.best_score_, rf_grid_search.best_params_, rf_grid_search.best_score_
 
class RandomForestWithForwardSelectionAndGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.rf = RandomForestClassifier()
        self.accuracy = None  # Stocke la précision

        # Définir la grille de paramètres pour la recherche en grille de Random Forest
        self.rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.rf, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def forward_selection(self, X, y, max_features):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]

        for num_features in range(1, max_features + 1):
            # Utiliser les 'num_features' premières colonnes comme caractéristiques
            X_selected = X[:, :num_features]

            # Ajuster le modèle RandomForest
            self.rf.fit(X_selected, y)

            # Effectuer la validation croisée
            scores = cross_val_score(self.rf, X_selected, y, cv=5, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected

        return best_X_selected, best_num_features, best_accuracy

    def fit(self, X, y):
        # Déterminer le meilleur nombre de caractéristiques à utiliser
        X_selected, best_num_features, best_accuracy = self.forward_selection(X, y, X.shape[1])

        # Ajuster le modèle RandomForest avec le meilleur nombre de caractéristiques
        self.rf.fit(X_selected, y)
        self.best_num_features = best_num_features
        self.best_accuracy = best_accuracy
        
        self.train_accuracy = self.rf.score(X_selected, y)

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_accuracy(self):
        return self.best_accuracy

    def get_best_num_features(self):
        return self.best_num_features
    
    def plot_learning_curve(self, X, y,train_sizes,cv=None):
        if cv is None:
            cv = self.find_best_num_folds(X, y, cv_range)  # Use the best number of folds if num_folds is not provided
        train_sizes, train_scores, test_scores = learning_curve(
            self.rf, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("Random Forest")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):  # Permet de spécifier le nombre de plis
        if num_folds is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5])  
            num_folds = best_num_folds
        # Recherche en grille pour RandomForest
        rf_grid_search = GridSearchCV(self.rf, self.rf_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        rf_grid_search.fit(X, y)
        best_rf = rf_grid_search.best_estimator_

        return best_rf, rf_grid_search.best_params_, rf_grid_search.best_score_
    
    def predict(self, X):
        return self.rf.predict(X)

