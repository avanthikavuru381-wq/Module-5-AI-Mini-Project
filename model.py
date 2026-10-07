# AI-Powered Student Performance Predictor
# Module 5 - AI Tools & Mini Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_model(data):
    """
    Train a Random Forest model to predict student performance.
    """

    features = [
        "study_hours",
        "attendance",
        "previous_score",
        "sleep_hours"
    ]

    target = "final_score"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    return model, mae