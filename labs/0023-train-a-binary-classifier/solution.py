# --- cell 1 ---
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train(X_train, y_train, X_val, y_val):

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    # Optional validation evaluation
    val_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, val_pred))

    def predict(X):
        return model.predict(X).astype(int)

    return predict