import re
import pandas as pd
import numpy as np
from pymatgen.ext.matproj import MPRester
from collections import defaultdict
from mendeleev import get_all_elements

def get_element_properties():
    elements = get_all_elements()
    properties = {}
    for element in elements:
        props = [element.atomic_number, element.covalent_radius, element.en_pauling, element.electron_affinity]
        ionization = list(element.ionenergies.values())[0] if element.ionenergies else None
        if None not in props and ionization is not None:
            props.append(ionization)
            properties[element.symbol] = props
    return properties

df = pd.read_csv('bond_matrix.csv')

API_KEY = 'd03v2DppwYYUcD6lhDB6MftdPma56l8A'

composition, theoretical = [], []

with MPRester(API_KEY) as mpr:
    for material_id in df['id']:
        material_data = mpr.materials.summary.search(material_ids=[material_id], fields=["formula_pretty", "theoretical"])
        if material_data:
            summary = material_data[0]
            composition.append(summary.formula_pretty)
            theoretical.append(summary.theoretical)
        else:
            composition.append(None)
            theoretical.append(None)

df['formula'] = composition
df['theoretical'] = theoretical


def multiply_elements(elements_dict, multiplier):
    return {el: count * multiplier for el, count in elements_dict.items()}

def extract_elements(formula):
    def parse_formula(formula):
        pattern = re.compile(r'([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)')
        stack = [defaultdict(int)]
        for elem, count, open_paren, close_paren, group_count in pattern.findall(formula):
            if elem:
                stack[-1][elem] += int(count) if count else 1
            elif open_paren:
                stack.append(defaultdict(int))
            elif close_paren:
                top = stack.pop()
                multiplier = int(group_count) if group_count else 1
                for el, cnt in top.items():
                    stack[-1][el] += cnt * multiplier
        return stack[0]
    return dict(parse_formula(formula))

df['formula'] = df['formula'].astype(str)
df['Elements'] = df['formula'].apply(extract_elements)
df_elements = pd.DataFrame(df['Elements'].tolist()).fillna(0)
df = df.drop(columns=['Elements'])

properties = get_element_properties()

media_features = {
    'nuclear_charge': [],
    'covalent_radius': [],
    'electronegativity': [],
    'electron_affinity': [],
    'ionization_energy': []
}

for _, row in df_elements.iterrows():
    total = row.sum()
    props_sum = dict.fromkeys(media_features.keys(), 0)
    for element, quantity in row.items():
        if quantity > 0 and element in properties:
            values = properties[element]
            for i, key in enumerate(media_features.keys()):
                props_sum[key] += (quantity * values[i]) / total
    for key in media_features:
        media_features[key].append(props_sum[key])

for key in media_features:
    df[key] = media_features[key]

minmax_features = {
    'nuclear_charge': ([], []),
    'covalent_radius': ([], []),
    'electronegativity': ([], []),
    'electron_affinity': ([], []),
    'ionization_energy': ([], [])
}

for _, row in df_elements.iterrows():
    values = {key: [] for key in minmax_features}
    for element, quantity in row.items():
        if quantity != 0 and element in properties:
            for i, key in enumerate(minmax_features):
                values[key].append(properties[element][i])
    for key in minmax_features:
        vals = sorted([v for v in values[key] if v is not None])
        minmax_features[key][0].append(vals[0] if vals else None)
        minmax_features[key][1].append(vals[-1] if vals else None)

for key, (min_list, max_list) in minmax_features.items():
    df[f'{key}_min'] = min_list
    df[f'{key}_max'] = max_list

properties_pairs = properties

columns_to_remove = ['nuclear_charge', 'covalent_radius', 'electronegativity', 'electron_affinity', 'nuclear_charge_max', 'nuclear_charge_min', 
    'covalent_radius_max', 'covalent_radius_min', 'electronegativity_max', 'electronegativity_min', 'electron_affinity_max', 
    'electron_affinity_min', 'ionization_energy', 'ionization_energy_max', 'ionization_energy_min', 'formula', 'id', 'theoretical']

df2 = df.drop(columns=columns_to_remove)

binary_features = {
    'Cr': 0,
    'E': 1,
    'EA': 2,
    'I': 3
}

def apply_binary_operations(df2, properties, binary_features):
    result = {f'{prefix}{op}': [] for prefix in binary_features for op in ['+', '-', '*', '÷']}

    for _, row in df2.iterrows():
        values = {key: {'+': 0, '-': 0, '*': 0, '÷': 0} for key in binary_features}
        for column, number in row.items():
            if number != 0:
                match = re.match(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", column)
                if not match:
                    continue
                a, b = match.groups()
                if a in properties and b in properties:
                    for key, idx in binary_features.items():
                        x, y = properties[a][idx], properties[b][idx]
                        values[key]['+'] += number * (x + y)
                        values[key]['-'] += number * (x - y)
                        values[key]['*'] += number * (x * y)
                        values[key]['÷'] += number * (x / y)
        for key in binary_features:
            for op in ['+', '-', '*', '÷']:
                result[f'{key}{op}'].append(values[key][op])

    return result

binary_result = apply_binary_operations(df2, properties_pairs, binary_features)

df = df.assign(**binary_result)

df.to_csv('material_data.csv', index=False)