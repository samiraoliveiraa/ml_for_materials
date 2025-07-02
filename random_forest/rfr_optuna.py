import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from joblib import dump

# Data
df = pd.read_csv('material_data.csv')

TEST_SIZE = 0.1
RANDOM_SEED = 24

FEATURES = [
    'nuclear_charge', 'covalent_radius', 'electronegativity', 'electron_affinity',
    'nuclear_charge_max', 'nuclear_charge_min', 'covalent_radius_max', 'covalent_radius_min',
    'electronegativity_max', 'electronegativity_min', 'electron_affinity_max', 'electron_affinity_min',
    'ionization_energy', 'ionization_energy_max', 'ionization_energy_min',
    'Cr+', 'Cr-', 'Cr*', 'Cr÷', 'E+', 'E-', 'E*', 'E÷',
    'EA+', 'EA-', 'EA*', 'EA÷', 'I+', 'I-', 'I*', 'I÷'
]

TARGET = ['ef']

df = df.reindex(FEATURES + TARGET, axis=1)
df = df.dropna()

X = df.reindex(FEATURES, axis=1)
y = df.reindex(TARGET, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "poisson"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        "max_features": trial.suggest_float("max_features_internal", 0, 1),
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train.values.ravel())
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best hyperparameters:", best_params)

best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train.values.ravel())

dump(best_model, 'rfr_model_optimized.joblib')
