import re
import pandas as pd
import numpy as np
from pymatgen.ext.matproj import MPRester
from collections import defaultdict
from mendeleev import get_all_elements

df = pd.read_csv('matriz_ligacoes.csv')

API_KEY = 'd03v2DppwYYUcD6lhDB6MftdPma56l8A'

composition = []
theoretical = []

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
            if elem:  # element with optional count
                stack[-1][elem] += int(count) if count else 1
            elif open_paren:  # opening parenthesis
                stack.append(defaultdict(int))
            elif close_paren:  # closing parenthesis
                top = stack.pop()
                multiplier = int(group_count) if group_count else 1
                for el, cnt in top.items():
                    stack[-1][el] += cnt * multiplier
        
        return stack[0]
    
    return dict(parse_formula(formula))

df['formula'] = df['formula'].astype(str)
df['Elementos'] = df['formula'].apply(extract_elements)
df_elementos = pd.DataFrame(df['Elementos'].tolist()).fillna(0)
df = df.drop(columns=['Elementos'])

elementos = get_all_elements()

propriedades_ionizacao = {}

for elemento in elementos:
    propriedades_ionizacao[elemento.symbol] = elemento.ionenergies

propriedades_ionizacao = {elemento: list(ionizacoes.values())[0] if ionizacoes.values() else None for elemento, ionizacoes in propriedades_ionizacao.items()}

propriedades = {}

for elemento in elementos:
    propriedades[elemento.symbol] = [elemento.atomic_number, elemento.covalent_radius, elemento.en_pauling, elemento.electron_affinity]
    
for elemento, valor in propriedades_ionizacao.items():
    if elemento in propriedades:
        propriedades[elemento].append(valor)
        
propriedades = {chave: tuple(valor) for chave, valor in propriedades.items()}

# propriedades = (número atômico, raio covalente, eletronegatividade, afinidade eletrônica, energia de ionização)

nuclear_charge_molecula = []
covalent_radius_molecula = []
electronegativity_molecula = []
electron_affinity_molecula = []
ionization_energy_molecula = []


for indice, linha in df_elementos.iterrows():
    nuclear_charge = 0
    covalent_radius = 0
    electronegativity = 0
    electron_affinity = 0
    ionization_energy = 0
    
    soma_elementos = linha.sum()

    for elemento, quantidade in linha.items():
        if quantidade > 0:
            parametros = propriedades.get(elemento, (None, None, None, None, None))
            if parametros[0] is not None:
                nuclear_charge += (quantidade * parametros[0]) / soma_elementos
            if parametros[1] is not None:
                covalent_radius += (quantidade * parametros[1]) / soma_elementos
            if parametros[2] is not None:
                electronegativity += (quantidade * parametros[2]) / soma_elementos
            if parametros[3] is not None:
                electron_affinity += (quantidade * parametros[3]) / soma_elementos
            if parametros[4] is not None:
                ionization_energy += (quantidade * parametros[4]) / soma_elementos


    nuclear_charge_molecula.append(nuclear_charge)
    covalent_radius_molecula.append(covalent_radius)
    electronegativity_molecula.append(electronegativity)
    electron_affinity_molecula.append(electron_affinity)
    ionization_energy_molecula.append(ionization_energy)


df['nuclear_charge'] = nuclear_charge_molecula
df['covalent_radius'] = covalent_radius_molecula
df['electronegativity'] = electronegativity_molecula
df['electron_affinity'] = electron_affinity_molecula
df['ionization_energy'] = ionization_energy_molecula

nuclear_charge_max = []
nuclear_charge_min = []
covalent_radius_max = []
covalent_radius_min = []
electronegativity_max = []
electronegativity_min = []
electron_affinity_max = []
electron_affinity_min = []
ionization_energy_max = []
ionization_energy_min = []

for index, row in df_elementos.iterrows():
    
    nuclear_charges = []
    covalent_radius = []
    electronegativities = []
    electron_affinities = []
    ionization_energies = []
    
    for elemento, quantidade in row.items():
        if quantidade != 0: 

            propriedades_elemento = propriedades[elemento]
            nuclear_charges.append(propriedades_elemento[0])
            covalent_radius.append(propriedades_elemento[1])
            electronegativities.append(propriedades_elemento[2])
            electron_affinities.append(propriedades_elemento[3])
            ionization_energies.append(propriedades_elemento[4])
    
    nuclear_charges = [x for x in nuclear_charges if x is not None]
    covalent_radius = [x for x in covalent_radius if x is not None]
    electronegativities = [x for x in electronegativities if x is not None]
    electron_affinities = [x for x in electron_affinities if x is not None]
    ionization_energies = [x for x in ionization_energies if x is not None]
    
    nuclear_charges.sort()
    covalent_radius.sort()
    electronegativities.sort()
    electron_affinities.sort()
    ionization_energies.sort()
    
    nuclear_charge_min.append(nuclear_charges[0] if nuclear_charges else None)
    nuclear_charge_max.append(nuclear_charges[-1] if nuclear_charges else None)
    
    covalent_radius_min.append(covalent_radius[0] if covalent_radius else None)
    covalent_radius_max.append(covalent_radius[-1] if covalent_radius else None)
    
    electronegativity_min.append(electronegativities[0] if electronegativities else None)
    electronegativity_max.append(electronegativities[-1] if electronegativities else None)
    
    electron_affinity_min.append(electron_affinities[0] if electron_affinities else None)
    electron_affinity_max.append(electron_affinities[-1] if electron_affinities else None)
    
    ionization_energy_min.append(ionization_energies[0] if ionization_energies else None)
    ionization_energy_max.append(ionization_energies[-1] if ionization_energies else None)
    

df['nuclear_charge_max'] = nuclear_charge_max
df['nuclear_charge_min'] = nuclear_charge_min
df['covalent_radius_max'] = covalent_radius_max
df['covalent_radius_min'] = covalent_radius_min
df['electronegativity_max'] = electronegativity_max
df['electronegativity_min'] = electronegativity_min
df['electron_affinity_max'] = electron_affinity_max
df['electron_affinity_min'] = electron_affinity_min
df['ionization_energy_max'] = ionization_energy_max
df['ionization_energy_min'] = ionization_energy_min

all_elements = get_all_elements()

properties = {}

for element in all_elements:
    properties[element.symbol] = [element.atomic_number, element.covalent_radius, element.en_pauling, element.electron_affinity]
    
ionization_dic = {}

for element in all_elements:
    ionization_dic[element.symbol] = element.ionenergies
    
ionization_dic = {elemento: list(ionizacoes.values())[0] if ionizacoes else None for elemento, ionizacoes in ionization_dic.items()}
    
for elemento in properties.keys():
    if elemento in ionization_dic and ionization_dic[elemento]:
        properties[elemento] += [ionization_dic[elemento]] 
        
properties = {k: v for k, v in properties.items() if None not in v}

df2 = df.drop(columns=['nuclear_charge', 'covalent_radius', 'electronegativity', 'electron_affinity', 'nuclear_charge_max', 'nuclear_charge_min', 
                      'covalent_radius_max', 'covalent_radius_min', 'electronegativity_max', 'electronegativity_min', 'electron_affinity_max', 
                      'electron_affinity_min', 'ionization_energy', 'ionization_energy_max', 'ionization_energy_min', 'formula', 'id', 'theoretical'])


list_covalent_radius_plus = []
list_covalent_radius_minus = []
list_covalent_radius_times = []
list_covalent_radius_divided = []

list_eletronegativity_plus = []
list_eletronegativity_minus = []
list_eletronegativity_times = []
list_eletronegativity_divided = []

list_electron_affinity_plus = []
list_electron_affinity_minus = []
list_electron_affinity_times = []
list_electron_affinity_divided = []

list_ionization_energy_plus = []
list_ionization_energy_minus = []
list_ionization_energy_times = []
list_ionization_energy_divided = []

for index, row in df2.iterrows():
    
    covalent_radius_plus = 0
    covalent_radius_minus = 0
    covalent_radius_times = 0
    covalent_radius_divided = 0
    
    eletronegativity_plus = 0
    eletronegativity_minus = 0
    eletronegativity_times = 0
    eletronegativity_divided = 0
    
    electron_affinity_plus = 0
    electron_affinity_minus = 0
    electron_affinity_times = 0
    electron_affinity_divided = 0
    
    ionization_energy_plus = 0
    ionization_energy_minus = 0
    ionization_energy_times = 0
    ionization_energy_divided = 0

    for coluna, numero in row.items():
        if numero != 0: 

            match = re.match(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", coluna)
            first_element = match.group(1)
            second_element = match.group(2)
            
            if first_element in properties and second_element in properties:

                covalent_radius_plus += numero * (properties[first_element][0] + properties[second_element][0])
                covalent_radius_minus += numero * (properties[first_element][0] - properties[second_element][0])
                covalent_radius_times += numero * (properties[first_element][0] * properties[second_element][0])
                covalent_radius_divided += numero * (properties[first_element][0] / properties[second_element][0])

                eletronegativity_plus += numero * (properties[first_element][1] + properties[second_element][1])
                eletronegativity_minus += numero * (properties[first_element][1] - properties[second_element][1])
                eletronegativity_times += numero * (properties[first_element][1] * properties[second_element][1])
                eletronegativity_divided += numero * (properties[first_element][1] / properties[second_element][1])

                electron_affinity_plus += numero * (properties[first_element][2] + properties[second_element][2])
                electron_affinity_minus += numero * (properties[first_element][2] - properties[second_element][2])
                electron_affinity_times += numero * (properties[first_element][2] * properties[second_element][2])
                electron_affinity_divided += numero * (properties[first_element][2] / properties[second_element][2])

                ionization_energy_plus += numero * (properties[first_element][3] + properties[second_element][3])
                ionization_energy_minus += numero * (properties[first_element][3] - properties[second_element][3])
                ionization_energy_times += numero * (properties[first_element][3] * properties[second_element][3])
                ionization_energy_divided += numero * (properties[first_element][3] / properties[second_element][3])
    
    
    list_covalent_radius_plus.append(covalent_radius_plus)
    list_covalent_radius_minus.append(covalent_radius_minus)
    list_covalent_radius_times.append(covalent_radius_times)
    list_covalent_radius_divided.append(covalent_radius_divided)
    
    list_eletronegativity_plus.append(eletronegativity_plus)
    list_eletronegativity_minus.append(eletronegativity_minus)
    list_eletronegativity_times.append(eletronegativity_times)
    list_eletronegativity_divided.append(eletronegativity_divided)

    list_electron_affinity_plus.append(electron_affinity_plus)
    list_electron_affinity_minus.append(electron_affinity_minus)
    list_electron_affinity_times.append(electron_affinity_times)
    list_electron_affinity_divided.append(electron_affinity_divided)

    list_ionization_energy_plus.append(ionization_energy_plus)
    list_ionization_energy_minus.append(ionization_energy_minus)
    list_ionization_energy_times.append(ionization_energy_times)
    list_ionization_energy_divided.append(ionization_energy_divided)
    
    
df['Cr+'] = list_covalent_radius_plus
df['Cr-'] = list_covalent_radius_minus
df['Cr*'] = list_covalent_radius_times
df['Cr÷'] = list_covalent_radius_divided

df['E+'] = list_eletronegativity_plus
df['E-'] = list_eletronegativity_minus
df['E*'] = list_eletronegativity_times
df['E÷'] = list_eletronegativity_divided

df['EA+'] = list_electron_affinity_plus
df['EA-'] = list_electron_affinity_minus
df['EA*'] = list_electron_affinity_times
df['EA÷'] = list_electron_affinity_divided

df['I+'] = list_ionization_energy_plus
df['I-'] = list_ionization_energy_minus
df['I*'] = list_ionization_energy_times
df['I÷'] = list_ionization_energy_divided

df.to_csv('material_data.csv', index=False)