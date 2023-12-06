import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LDAWithRegularization:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.lda = LinearDiscriminantAnalysis()
        self.accuracy_train = None  # Store training accuracy
        self.accuracy_test = None  # Store test accuracy

        # Define the grid of parameters for LDA with regularization
        self.lda_param_grid = {
            'solver': ['lsqr'],
            'shrinkage': ['auto', 0.1, 0.5, 0.9],  # Regularization parameter
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.lda, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def find_best_num_folds_default(self, X, y):
        return self.find_best_num_folds(X, y, cv_range=[3, 5, 7])

    def fit(self, X_train, y_train, X_test, y_test, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds_default(X_train, y_train)
            cv = best_num_folds

        # Apply feature scaling (standardization) on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Grid search for LDA on training data
        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X_train_scaled, y_train)
        best_lda = lda_grid_search.best_estimator_

        # Use the pipeline for cross-validation on training data
        scores_train = cross_val_score(best_lda, X_train_scaled, y_train, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy_train = scores_train.mean()

        # Apply the same feature scaling on test data
        X_test_scaled = scaler.transform(X_test)

        # Evaluate the model on test data
        self.accuracy_test = accuracy_score(y_test, best_lda.predict(X_test_scaled))

    def get_accuracy_train(self):
        return self.accuracy_train

    def get_accuracy_test(self):
        return self.accuracy_test

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.lda, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("LDA")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds_default(X, y)

        # Grid search for LDA
        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X, y)
        best_lda = lda_grid_search.best_estimator_

        return best_lda, lda_grid_search.best_params_, lda_grid_search.best_score_

    def predict(self, X):
        return self.lda.predict(X)
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.lda = LinearDiscriminantAnalysis()
        self.accuracy_train = None  # Store training accuracy
        self.accuracy_test = None  # Store test accuracy

        # Define the grid of parameters for LDA with regularization
        self.lda_param_grid = {
            'solver': ['lsqr'],
            'shrinkage': ['auto', 0.1, 0.5, 0.9],  # Regularization parameter
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.lda, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def find_best_num_folds_default(self, X, y):
        return self.find_best_num_folds(X, y, cv_range=[3, 5, 7])

    def fit(self, X_train, y_train, X_test, y_test, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds_default(X_train, y_train)
            cv = best_num_folds

        # Apply feature scaling (standardization) on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Grid search for LDA on training data
        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X_train_scaled, y_train)
        best_lda = lda_grid_search.best_estimator_

        # Use the pipeline for cross-validation on training data
        scores_train = cross_val_score(best_lda, X_train_scaled, y_train, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy_train = scores_train.mean()

        # Apply the same feature scaling on test data
        X_test_scaled = scaler.transform(X_test)

        # Evaluate the model on test data
        self.accuracy_test = accuracy_score(y_test, best_lda.predict(X_test_scaled))

    def get_accuracy_train(self):
        return self.accuracy_train

    def get_accuracy_test(self):
        return self.accuracy_test

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.lda, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("LDA")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, num_folds=None):
        if num_folds is None:
            num_folds = self.find_best_num_folds_default(X, y)

        # Grid search for LDA
        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X, y)
        best_lda = lda_grid_search.best_estimator_

        return best_lda, lda_grid_search.best_params_, lda_grid_search.best_score_

    def predict(self, X):
        return self.lda.predict(X)

class LDAWithCrossValidationPCA:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.lda = LinearDiscriminantAnalysis()
        self.accuracy = None  # Store accuracy

        # Define the grid of parameters for LDA
        self.lda_param_grid = {
            'solver': ['lsqr'],
            'shrinkage': ['auto', 0.5, 1.0],
        }

        # Add PCA components as a range
        self.pca_components_range = [10, 20, 30, 40, 50]

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.lda, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_folds = num_folds

        return best_num_folds

    def find_best_pca_components(self, X, y):
        best_components = None
        best_accuracy = 0.0

        for n_components in self.pca_components_range:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)

            # Find the best number of folds
            best_num_folds = self.find_best_num_folds(X_pca, y, cv_range=[3, 5, 7])

            # Grid search for LDA
            lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
            lda_grid_search.fit(X_pca, y)
            accuracy = lda_grid_search.best_score_

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_components = n_components

        return best_components, best_num_folds

    def fit(self, X, y):
        # Find the best number of PCA components
        best_components, best_num_folds = self.find_best_pca_components(X, y)

        # Apply PCA with the best number of components
        pca = PCA(n_components=best_components)
        X_pca = pca.fit_transform(X)

        # Grid search for LDA
        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X_pca, y)
        best_lda = lda_grid_search.best_estimator_

        # Use the pipeline for cross-validation
        scores = cross_val_score(best_lda, X_pca, y, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        self.accuracy = scores.mean()

        # Fit the model on the full data
        best_lda.fit(X_pca, y)

        
    def get_accuracy(self):
        return self.accuracy

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.lda, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("LDA")
        plt.legend()
        plt.grid()

    def grid_search(self, X, y, n_components=None):
        # Apply PCA with the specified number of components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Grid search for LDA
        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X_pca, y)
        best_lda = lda_grid_search.best_estimator_

        return best_lda, lda_grid_search.best_params_, lda_grid_search.best_score_
    
    def predict(self, X):
        return self.lda.predict(X)

class LDAWithForwardSelectionAndGridSearch:
    def __init__(self, n_jobs=-1, scoring="accuracy"):
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.lda = LinearDiscriminantAnalysis()
        self.scaler = StandardScaler()  # Add StandardScaler
        self.accuracy = None
        self.lda_param_grid = {
            'solver': ['lsqr'],
            'shrinkage': ['auto', 0.5, 1.0], 
        }

    def find_best_num_folds(self, X, y, cv_range):
        best_num_folds = None
        best_accuracy = 0.0

        for num_folds in cv_range:
            scores = cross_val_score(self.lda, X, y, cv=num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
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
            X_selected = X[:, :num_features]
            
            # Apply the same feature scaling as during training
            X_selected_scaled = self.scaler.transform(X_selected)

            self.lda.fit(X_selected_scaled, y)
            scores = cross_val_score(self.lda, X_selected_scaled, y, cv=self.find_best_num_folds(X, y, cv_range=[3, 5, 7]), n_jobs=self.n_jobs, scoring=self.scoring)
            accuracy = scores.mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_num_features = num_features
                best_X_selected = X_selected_scaled

        return best_X_selected, best_num_features, best_accuracy


    def fit(self, X, y):
        X_selected, best_num_features, best_accuracy = self.forward_selection(X, y, X.shape[1])
        self.lda.fit(X_selected, y)
        self.best_num_features = best_num_features
        self.best_accuracy = best_accuracy

    def get_accuracy(self):
        return self.best_accuracy

    def get_best_num_features(self):
        return self.best_num_features

    def plot_learning_curve(self, X, y, train_sizes, cv=None):
        if cv is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
            cv = best_num_folds

        train_sizes, train_scores, test_scores = learning_curve(
            self.lda, X, y, train_sizes=train_sizes, cv=cv, n_jobs=self.n_jobs, scoring=self.scoring
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
        plt.title("LDA")
        plt.legend()
        plt.grid()        

    def grid_search(self, X, y, num_folds=None):
        if num_folds is None:
            best_num_folds = self.find_best_num_folds(X, y, cv_range=[3, 5, 7])
        else:
            best_num_folds = num_folds

        self.scaler.fit(X)  # Fit the scaler on the entire training set
        X_scaled = self.scaler.transform(X)

        lda_grid_search = GridSearchCV(self.lda, self.lda_param_grid, cv=best_num_folds, n_jobs=self.n_jobs, scoring=self.scoring)
        lda_grid_search.fit(X_scaled, y)
        best_lda = lda_grid_search.best_estimator_

        return best_lda, lda_grid_search.best_params_, lda_grid_search.best_score_

    def predict(self, X):
        # Apply the same feature scaling as during training
        X_scaled = self.scaler.transform(X)
        return self.lda.predict(X_scaled)


