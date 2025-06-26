import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class OnlineClassifier:
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_xy(self, df, feature_cols, target_col):
        X = df[feature_cols].values
        y = df[target_col].values
        return X, y

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def partial_fit(self, X, y):
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.model.partial_fit(X_scaled, y, classes=np.array([0, 1]))
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled) 