import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
 
class LogisticRegressionWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [5, 10, 15]  # Default CV range
        self.logreg = LogisticRegression()
        self.scaler = StandardScaler()  
        self.accuracy = None
        self.best_num_folds = None  
 
        # Définir les paramètres de la recherche par grille
        self.param_grid = {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0],  
        }
 
    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        self.best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return self.best_num_folds
 
    def fit(self, X, y, cv=1):
        self.best_num_folds = self.find_best_num_folds(X, y)
 
        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)
 
        scores = cross_val_score(self.logreg, X_scaled, y, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)
 
    def get_accuracy(self):
        return self.accuracy
 
    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            cv = self.best_num_folds  # Use the best number of folds if not provided
 
        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)
 
        train_sizes, train_scores, test_scores = learning_curve(
            self.logreg, X_scaled, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("Logistic Regression")
        plt.legend()
        plt.grid()
 
    def grid_search(self, X, y, cv=None):
        if cv is None:
            cv = self.best_num_folds  # Use the best number of folds if not provided
 
        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)
 
        grid_search = GridSearchCV(self.logreg, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        grid_search.fit(X_scaled, y)
        self.logreg = grid_search.best_estimator_
 
        return grid_search.best_params_, grid_search.best_score_
 
class PCAandLogisticRegressionWithCrossValidationGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [5, 10, 15]  
        self.pca = PCA()
        self.logreg = LogisticRegression()
        self.scaler = StandardScaler()  
        self.accuracy = None  
        self.best_num_folds = None  
 
        # Définir la grille de paramètres pour la recherche de grille
        self.pca_param_grid = {
            'n_components': [100, 120, 130, 140, 160],
        }
 
        self.logreg_param_grid = {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0], 
        }
 
    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        self.best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return self.best_num_folds
 
    def fit(self, X, y, cv=1):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
 
        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)
 
        scores = cross_val_score(self.logreg, X_scaled, y, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = np.mean(scores)
 
    def get_accuracy(self):
        return self.accuracy
 
    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            cv = self.best_num_folds  
 
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
 
        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)
 
        train_sizes, train_scores, test_scores = learning_curve(
            self.logreg, X_scaled, y, train_sizes=train_sizes, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Tracer les résultats moyens d'entrainement et des tests
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats d'entrainement et de test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Logistic Regression")
        plt.legend()
        plt.grid()
 
    def grid_search(self, X, y, cv=None):
        if cv is None:
            cv = self.best_num_folds  
 
        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)
 
        # Grid search pour PCA
        pca_grid_search = GridSearchCV(self.pca, self.pca_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        pca_grid_search.fit(X_scaled, y)
        best_pca = pca_grid_search.best_estimator_
 
        # Appliquer PCA transformation
        X_pca = best_pca.transform(X_scaled)
 
        # Grid search for Logistic Regression
        logreg_grid_search = GridSearchCV(self.logreg, self.logreg_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        logreg_grid_search.fit(X_pca, y)
        best_logreg = logreg_grid_search.best_estimator_
 
        return best_pca, best_logreg, pca_grid_search.best_params_, pca_grid_search.best_score_, logreg_grid_search.best_params_, logreg_grid_search.best_score_
 
class LogisticRegressionWithForwardSelection:
    def __init__(self, n_jobs=-1, scoring="accuracy", cv_range=None):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv_range = cv_range if cv_range is not None else [5, 10, 15]  # Default CV range
        self.logreg = LogisticRegression()
        self.scaler = StandardScaler()  # Add StandardScaler
        self.accuracy = None
        self.best_num_folds = None
        self.best_num_features = None  # Added for forward selection

    def find_best_num_folds(self, X, y):
        min_samples_per_class = min(np.bincount(y))
        self.best_num_folds = min(min_samples_per_class, max(self.cv_range))
        return self.best_num_folds

    def forward_selection(self, X, y):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]

        for num_features in range(1, X.shape[1] + 1):
            # Use the first 'num_features' columns as features
            X_selected = X[:, :num_features]

            # Appliquer StandardScaler
            X_selected_scaled = self.scaler.fit_transform(X_selected)

            # Fit the logistic regression model
            self.logreg.fit(X_selected_scaled, y)

            # Perform cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                self.logreg, X_selected_scaled, y, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring
            )
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected

        return best_X_selected, best_num_features, best_accuracy

    def fit(self, X, y, cv=1):
        self.best_num_folds = self.find_best_num_folds(X, y)

        # Forward selection
        X_selected, self.best_num_features, best_accuracy = self.forward_selection(X, y)

        # Appliquer StandardScaler to the selected features
        X_selected_scaled = self.scaler.fit_transform(X_selected)

        # Ajuster le modèle de régression logistique
        self.logreg.fit(X_selected_scaled, y)
        self.accuracy = best_accuracy

    def get_accuracy(self):
        return self.accuracy

    def get_best_num_features(self):
        return self.best_num_features

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            cv = self.best_num_folds  

        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)

        train_sizes, train_scores, test_scores = learning_curve(
            self.logreg, X_scaled, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)


        plt.figure(figsize=(10, 6))

        # Tracer les résultats moyens d'entrainement et des tests
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats d'entrainement et de test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Logistic Regression")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, cv=None):
        if cv is None:
            cv = self.best_num_folds  

        # Appliquer StandardScaler
        X_scaled = self.scaler.fit_transform(X)

        # Grid search pour la Regression Logistique
        logreg_param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]}
        grid_search = GridSearchCV(
            self.logreg, logreg_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
        )
        grid_search.fit(X_scaled, y)
        self.logreg = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_
