from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector


class KNNWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.knn = KNeighborsClassifier()
        self.cv_range = cv_range if cv_range is not None else [3, 5, 7]
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.scaler = StandardScaler()

        # Define the parameter grid for the grid search
        self.knn_param_grid = {
            'n_neighbors': [3, 5, 7, 10],  # Add more values for n_neighbors
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.knn, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, self.cv_range)
        X_scaled = self.scaler.fit_transform(X)
        
        # Ajustez le modèle sur l'ensemble de l'entraînement pour obtenir la précision de l'entraînement
        self.knn.fit(X_scaled, y)
        self.train_accuracy = self.knn.score(X_scaled, y)

        # Cross-validation scores
        scores = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = scores.mean()
        self.precision = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='precision_macro').mean()
        self.recall = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='recall_macro').mean()
        self.f1 = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='f1_macro').mean()

    def get_precision(self):
        return self.precision

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return self.f1

    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            cv = self.find_best_num_folds(X, y, self.cv_range)
        X_scaled = self.scaler.fit_transform(X)

        train_sizes, train_scores, test_scores = learning_curve(
            self.knn, X_scaled, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("KNN")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, self.cv_range)

        # Grid search for KNN
        knn_grid_search = GridSearchCV(self.knn, self.knn_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        knn_grid_search.fit(X, y)
        best_knn = knn_grid_search.best_estimator_

        # Store the best instance of KNN
        self.knn = best_knn

        return best_knn, knn_grid_search.best_params_, knn_grid_search.best_score_

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.knn.predict(X_scaled)

class PCAandKNNWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.pca = PCA()
        self.knn = KNeighborsClassifier()
        self.cv_range = cv_range if cv_range is not None else [3, 5, 7] 
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.scaler = StandardScaler()

        # Définition de la grille des paramètres pour la recherche par grille de PCA
        self.pca_param_grid = {
            'n_components': [100, 120, 130, 140, 160],
        }
       # Définition de la grille des paramètres pour la recherche par grille de KNN
        self.knn_param_grid = {
            'n_neighbors': [3, 5, 7, 10],  # Add more values for n_neighbors
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.knn, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def fit(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, cv_range)
        X_scaled = self.scaler.fit_transform(X)
        # Fit PCA and KNN on the entire training set to get training accuracy
        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)
        self.knn.fit(X_pca, y)
        self.train_accuracy = self.knn.score(X_pca, y) 
        
        scores = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = scores.mean()
        self.precision = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='precision_macro').mean()
        self.recall = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='recall_macro').mean()
        self.f1 = cross_val_score(self.knn, X_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='f1_macro').mean()

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return self.f1

    def get_accuracy(self):
        return self.accuracy
    
    def get_train_accuracy(self):
        return self.train_accuracy
    
    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            cv = self.find_best_num_folds(X, y, self.cv_range)
        X_scaled = self.scaler.fit_transform(X)

        train_sizes, train_scores, test_scores = learning_curve(
            self.knn, X_scaled, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("KNN")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, cv_range)
        pca_grid_search = GridSearchCV(self.pca, self.pca_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        pca_grid_search.fit(X, y)
        best_pca = pca_grid_search.best_estimator_

        # Recherche par grille pour KNN avec les données transformées par PCA
        knn_grid_search = GridSearchCV(self.knn, self.knn_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        knn_grid_search.fit(best_pca.transform(X), y)
        best_knn = knn_grid_search.best_estimator_

        # Stocke les meilleures instances de PCA et de KNN
        self.pca = best_pca
        self.knn = best_knn

        return best_pca, best_knn, pca_grid_search.best_params_, pca_grid_search.best_score_, knn_grid_search.best_params_, knn_grid_search.best_score_

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.knn.predict(X_scaled)

class KNNWithForwardSelection:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.knn = KNeighborsClassifier()
        self.cv_range = cv_range if cv_range is not None else [3, 5, 7]
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.scaler = StandardScaler()
        self.best_num_folds = None
        self.best_num_features = None  # Added for forward selection

        # Define the parameter grid for GridSearch
        self.knn_param_grid = {
            'n_neighbors': [3, 5, 7, 10],  # Add more values for n_neighbors
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.knn, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

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

            # Appliquer StandardScaler aux fonctionnalités sélectionnées
            X_selected_scaled = self.scaler.fit_transform(X_selected)

            # Adaptez le modèle KNN
            self.knn.fit(X_selected_scaled, y)

            # Performmer cross-validation
            scores = cross_val_score(self.knn, X_selected_scaled, y, cv=5, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected_scaled

        return best_X_selected, best_num_features, best_accuracy

    def fit(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, self.cv_range)

        # Forward selection
        X_selected, self.best_num_features, best_accuracy = self.forward_selection(X, y, X.shape[1])

        # Appliquer StandardScaler aux fonctionnalités sélectionnées
        X_selected_scaled = self.scaler.fit_transform(X_selected)

        # Adaptez le modèle KNN
        self.knn.fit(X_selected_scaled, y)
        self.accuracy = best_accuracy  # Store test accuracy
        self.precision = cross_val_score(
            self.knn, X_selected_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='precision_macro'
        ).mean()
        self.recall = cross_val_score(
            self.knn, X_selected_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='recall_macro'
        ).mean()
        self.f1 = cross_val_score(
            self.knn, X_selected_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring='f1_macro'
        ).mean()

        # Calculate and store train accuracy
        train_scores = cross_val_score(self.knn, X_selected_scaled, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        self.train_accuracy = train_scores.mean()

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return self.f1

    def get_accuracy(self):
        return self.accuracy

    def get_best_num_features(self):
        return self.best_num_features

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            cv = self.find_best_num_folds(X, y, self.cv_range)
        X_scaled = self.scaler.fit_transform(X)

        train_sizes, train_scores, test_scores = learning_curve(
            self.knn, X_scaled, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("KNN")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds(X, y, self.cv_range)

        # Appliquer StandardScaler à l'ensemble de données
        X_scaled = self.scaler.fit_transform(X)

        # Adaptez le modèle KNN
        knn_grid_search = GridSearchCV(
            self.knn, self.knn_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring
        )
        knn_grid_search.fit(X_scaled, y)
        best_knn = knn_grid_search.best_estimator_

        # Stockez la meilleure instance de KNN
        self.knn = best_knn

        return best_knn, knn_grid_search.best_params_, knn_grid_search.best_score_

    def predict(self, X):
        # Use the selected number of features
        X_selected = X[:, :self.best_num_features]

        # Apply StandardScaler to the selected features
        X_selected_scaled = self.scaler.transform(X_selected)

        return self.knn.predict(X_selected_scaled)
