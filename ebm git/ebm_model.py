from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump

df = pd.read_csv('material_data.csv')

TEST_SIZE = 0.1
RANDOM_SEED = 24

FEATURES = ['nuclear_charge', 'covalent_radius (pm)', 'electronegativity', 'electron_affinity (eV)', 'nuclear_charge_max', 'nuclear_charge_min', 'covalent_radius_max (pm)', 'covalent_radius_min (pm)', 'electronegativity_max', 'electronegativity_min', 'electron_affinity_max (eV)', 'electron_affinity_min (eV)', 'ionization_energy (eV)', 'ionization_energy_max (eV)', 'ionization_energy_min (eV)', 'Cr$_+$ (pm)', 'Cr$_-$ (pm)', 'Cr$_*$ (pm)', 'Cr$_÷$ (pm)', 'E$_+$', 'E$_-$', 'E$_*$', 'E$_÷$', 'EA$_+$ (eV)', 'EA$_-$ (eV)', 'EA$_*$ (eV)', 'EA$_÷$ (eV)', 'I$_+$ (eV)', 'I$_-$ (eV)', 'I$_*$ (eV)', 'I$_÷$ (eV)']

TARGET = ['ef (eV)']

df = df.reindex(FEATURES + TARGET, axis=1)
df = df.dropna() 

X = df.reindex(FEATURES, axis=1)
y = df.reindex(TARGET, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

ebm_model = ExplainableBoostingRegressor()
ebm_model.fit(X_train, y_train)

dump(ebm_model, 'ebm_model_trained.joblib')
