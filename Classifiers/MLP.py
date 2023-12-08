from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MLPClassifierCVGridSearch:
    def __init__(self, param_grid, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.clf = make_pipeline(StandardScaler(), MLPClassifier(activation='logistic', max_iter=3000))
        self.param_grid = param_grid
        self.cv_range = cv_range
 
    def grid_search(self, X, y):
        best_cv_score = -1
        best_cv = None
        best_params = None
 
        for cv in self.cv_range:
            grid_search = GridSearchCV(self.clf, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
            grid_search.fit(X, y)
 
            if grid_search.best_score_ > best_cv_score:
                best_cv_score = grid_search.best_score_
                best_cv = cv
                best_params = grid_search.best_params_
 
        self.clf = grid_search.best_estimator_
        return best_params, best_cv_score
 
    def plot_learning_curve(self, X, y, train_sizes):
        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Tracer les scores moyens de l'entrainement et du test
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats de l'entraînement et du test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Multi Layer Perceptron")
        plt.legend()
        plt.grid()
 
class MLPClassifierWithPCACVGridSearch:
    def __init__(self, param_grid, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.clf = make_pipeline(StandardScaler(), PCA(), MLPClassifier(activation='logistic', max_iter=3000))
        self.param_grid = param_grid
        self.cv_range = cv_range

    def grid_search(self, X, y):
        best_cv_score = -1
        best_cv = None
        best_params = None

        for cv in self.cv_range:
            grid_search = GridSearchCV(self.clf, self.param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
            grid_search.fit(X, y)

            if grid_search.best_score_ > best_cv_score:
                best_cv_score = grid_search.best_score_
                best_cv = cv
                best_params = grid_search.best_params_

        self.clf = grid_search.best_estimator_
        return best_params, best_cv_score


    def plot_learning_curve(self, X, y, train_sizes):
        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Tracer les scores moyens de l'entrainement et du test
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats de l'entraînement et du test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Multi Layer Perceptron")
        plt.legend()
        plt.grid()

class MLPClassifierCVFeatureSelection:
    def __init__(self, alpha_range, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.alpha_range = alpha_range
        self.cv_range = cv_range
        self.clf = None  # Stocke le meilleur estimateur
 
    def grid_search(self, X, y):
        best_cv_score = -1
        best_cv = None
        best_params = None
 
        for cv_outer in self.cv_range:
            for alpha in self.alpha_range:
                clf = MLPClassifier(alpha=alpha)
 
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
                    'activation': ['logistic', 'tanh', 'relu'],
                }
 
                skf = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
 
                grid_search = GridSearchCV(clf, param_grid, cv=skf, n_jobs=self.n_jobs, scoring=self.scoring)
                grid_search.fit(X, y)
 
                if grid_search.best_score_ > best_cv_score:
                    best_cv_score = grid_search.best_score_
                    best_cv = cv_outer
                    best_params = {
                        "alpha": alpha,
                        "architecture": grid_search.best_params_['hidden_layer_sizes'],
                        "activation": grid_search.best_params_['activation'],
                    }
                    self.clf = grid_search.best_estimator_
 
        return best_params, best_cv_score
   
    def plot_learning_curve(self, X, y, train_sizes):
        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))

        # Tracer les scores moyens de l'entrainement et du test
        plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color="r")
        plt.plot(train_sizes, test_scores_mean, label="Score de Test", color="g")

        # Ajouter des régions ombrées pour la variabilité des résultats de l'entraînement et du test
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.xlabel("Nombre d'échantillons d'entraînement")
        plt.ylabel('Accuracy')
        plt.title("Multi Layer Perceptron")
        plt.legend()
        plt.grid()
   
