import optuna
from interpret.glassbox import ExplainableBoostingRegressor
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
        "max_bins": trial.suggest_int("max_bins", 64, 512),
        "max_interaction_bins": trial.suggest_int("max_interaction_bins", 64, 256),
        "interactions": trial.suggest_int("interactions", 0, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_rounds": trial.suggest_int("max_rounds", 50, 500),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 5, 50),
        "n_jobs": -1,
        "random_state": RANDOM_SEED
    }

    model = ExplainableBoostingRegressor(**params)
    model.fit(X_train, y_train.values.ravel())
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best hyperparameters:", best_params)

best_model = ExplainableBoostingRegressor(**best_params)
best_model.fit(X_train, y_train.values.ravel())

dump(best_model, 'ebm_model_optimized.joblib')
