from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

        

class PerceptronCVGridSearch:
    def __init__(self, param_grid, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.clf = make_pipeline(StandardScaler(), Perceptron(eta0=0.001, penalty="l2", max_iter=3000))
        self.param_grid = param_grid
        self.cv_range = [3, 5, 7]

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

    def find_best_num_folds(self, X, y):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in self.cv_range:
            scores = cross_val_score(self.clf, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def plot_learning_curve(self, X, y, train_sizes, cv):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
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
        plt.title("Perceptron")
        plt.legend()
        plt.grid()

class PerceptronCVPCAWithPreprocessing:
    def __init__(self, alpha_range, pca_components_range, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.alpha_range = alpha_range
        self.pca_components_range = pca_components_range
        self.cv_range = [3, 5]
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('perceptron', Perceptron(max_iter=3000))
        ])

    def grid_search(self, X, y):
        param_grid = {
            'perceptron__alpha': self.alpha_range
        }

        best_score = -1
        best_params = None
        best_estimator = None

        for n_components in self.pca_components_range:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('perceptron', Perceptron(max_iter=3000))
            ])

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=self.n_jobs, scoring=self.scoring
            )
            grid_search.fit(X, y)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_params = {
                    "alpha": grid_search.best_params_['perceptron__alpha'],
                    "n_components": n_components,
                }
                best_estimator = grid_search.best_estimator_

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator

        return best_params, best_score
        
    def plot_learning_curve(self, X, y, train_sizes, cv):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y)
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.pipeline, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("Perceptron")
        plt.legend()
        plt.grid()
        
    def find_best_num_folds(self, X, y):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in self.cv_range:
            scores = cross_val_score(self.pipeline, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = np.mean(scores)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

class PerceptronCVForwardSelectionWithPreprocessing:
    def __init__(self, alpha_range, max_num_features_range, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.alpha_range = alpha_range
        self.max_num_features_range = max_num_features_range
        self.cv_range = cv_range
        self.clf = None  # Store the best estimator

    def forward_selection(self, X, y, max_features):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]
        best_params = None

        for num_features in range(1, max_features + 1):
            # Select the first `num_features` columns
            X_selected = X[:, :num_features]

            # Adjust the Perceptron model
            clf = Perceptron(max_iter=5000, eta0=0.1)  # Increase max_iter and try different values for eta0
            clf.fit(X_selected, y)

            # Perform cross-validation
            scores = cross_val_score(clf, X_selected, y, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected
                best_params = {"num_features": best_num_features, "accuracy": accuracy}

        #print(f"Best Parameters from forward_selection: {best_params}")  # Debug print
        return best_X_selected, best_num_features, best_params

    def grid_search(self, X, y):
        best_cv_score = -1
        best_params = None

        for cv in self.cv_range:
            for max_features in self.max_num_features_range:
                X_selected, best_num_features, _ = self.forward_selection(X, y, max_features)
                clf = Perceptron(max_iter=5000, eta0=0.1)  # Increase max_iter and try different values for eta0

                # Define a parameter grid for hyperparameter tuning
                param_grid = {
                    'alpha': self.alpha_range,
                }

                grid_search = GridSearchCV(clf, param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
                grid_search.fit(X_selected, y)

                if grid_search.best_score_ > best_cv_score:
                    best_cv_score = grid_search.best_score_
                    best_params = {
                        "max_features": best_num_features,
                        "alpha": grid_search.best_params_['alpha'],
                    }
                    self.clf = grid_search.best_estimator_  # Store the best estimator

        return best_params, best_cv_score

    def plot_learning_curve(self, X, y, train_sizes):
        if self.clf is None:
            print("Error: Classifier not initialized. Run grid_search method first.")
            return

        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("Perceptron")
        plt.legend()
        plt.grid()
        

    def __init__(self, alpha_range, max_num_features_range, cv_range, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.alpha_range = alpha_range
        self.max_num_features_range = max_num_features_range
        self.cv_range = cv_range
        self.clf = None  # Store the best estimator

    def forward_selection(self, X, y, max_features):
        best_accuracy = 0.0
        best_num_features = 1
        best_X_selected = X[:, :1]
        best_params = None

        for num_features in range(1, max_features + 1):
            # Select the first `num_features` columns
            X_selected = X[:, :num_features]

            # Adjust the Perceptron model
            clf = Perceptron(max_iter=5000, eta0=0.1)  # Increase max_iter and try different values for eta0
            clf.fit(X_selected, y)

            # Perform cross-validation
            scores = cross_val_score(clf, X_selected, y, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected
                best_params = {"num_features": best_num_features, "accuracy": accuracy}

        print(f"Best Parameters from forward_selection: {best_params}")  # Debug print
        return best_X_selected, best_num_features, best_params

    def grid_search(self, X, y):
        best_cv_score = -1
        best_params = None

        for cv in self.cv_range:
            for max_features in self.max_num_features_range:
                X_selected, best_num_features, _ = self.forward_selection(X, y, max_features)
                clf = Perceptron(max_iter=5000, eta0=0.1)  # Increase max_iter and try different values for eta0

                # Debug print
                print(f"CV: {cv}, Max Features: {max_features}")

                # Debug print
                print(f"Shape of X_selected: {X_selected.shape}")

                # Define a parameter grid for hyperparameter tuning
                param_grid = {
                    'alpha': self.alpha_range,
                }

                grid_search = GridSearchCV(clf, param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
                grid_search.fit(X_selected, y)

                # Debug print
                print(f"Grid Search Scores: {grid_search.cv_results_['mean_test_score']}")

                if grid_search.best_score_ > best_cv_score:
                    best_cv_score = grid_search.best_score_
                    best_params = {
                        "max_features": best_num_features,
                        "alpha": grid_search.best_params_['alpha'],
                    }
                    self.clf = grid_search.best_estimator_  # Store the best estimator

        return best_params, best_cv_score

    def plot_learning_curve(self, X, y, train_sizes):
        if self.clf is None:
            print("Error: Classifier not initialized. Run grid_search method first.")
            return

        train_sizes, train_scores, test_scores = learning_curve(
            self.clf, X, y, train_sizes=train_sizes, cv=self.cv_range[-1], n_jobs=self.n_jobs, scoring=self.scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label="Training Score")
        plt.plot(train_sizes, test_scores_mean, label="Test Score")
        plt.xlabel("Number of Training Samples")
        plt.ylabel('Accuracy')
        plt.title("Perceptron avec CV, Feature Selection")
        plt.legend()
        plt.grid()