import numpy as np
from pymatgen.core import Structure
from mendeleev import element 
import os
import pandas as pd
from periodictable import elements

def compute_atom_and_bond_vectors(poscar_file):
    delta = 0.3  # Sum of covalent radii + delta

    with open(poscar_file, 'r') as f:
        lines = f.readlines()
        if any('direct' in line.lower() for line in lines):
            coord_type = 'direct'
        elif any('cartesian' in line.lower() for line in lines):
            coord_type = 'cartesian'
        else:
            raise ValueError("Coordinate type not found in the POSCAR file.")

    structure = Structure.from_file(poscar_file)
    
    num_atoms = len(structure)
    atomic_numbers = [site.specie.Z for site in structure.sites]
    positions = np.array([site.frac_coords if coord_type == 'direct' else site.coords for site in structure.sites])
    lattice = structure.lattice.matrix
    if coord_type == 'direct':
        positions = np.dot(positions, lattice)  

    atom_vector = np.zeros(118, dtype=int)
    bond_vector = np.zeros(118 * 119 // 2, dtype=int) 

    for atomic_number in atomic_numbers:
        atom_vector[atomic_number - 1] += 1  

    covalent_radii = [element(site.specie.symbol).covalent_radius / 100 for site in structure.sites]

    def bond_index(z1, z2):
        return (z1 - 1) * 118 + (z2 - 1) - (z1 * (z1 - 1)) // 2

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    for m in [-1, 0, 1]:
                        translation = k * lattice[0] + l * lattice[1] + m * lattice[2]
                        distance_vector = positions[i] - (positions[j] + translation)
                        distance = np.linalg.norm(distance_vector)

                        sum_covalent_radii = covalent_radii[i] + covalent_radii[j] + delta
                        if distance <= sum_covalent_radii:
                            z1, z2 = sorted((atomic_numbers[i], atomic_numbers[j]))
                            bond_vector[bond_index(z1, z2)] += 1

    return atom_vector, bond_vector

directory = '/home/samira23016/IC/poscars_todos'
bond_matrix = []

for file in os.listdir(directory):
    if file.startswith("POSCAR") and file.endswith(".vasp"):
        print(f"Processando: {file}")  

        id_material = file.split('_')[1].split('.')[0]  
        file_path = os.path.join(directory, file)

        bond_vector = compute_atom_and_bond_vectors(file_path)[1]

        material_data = {'id': id_material}
        for i in range(len(bond_vector)):
            material_data[i] = bond_vector[i] 

        bond_matrix.append(material_data)

df = pd.DataFrame(bond_matrix)

atomic_symbols = [elements[i].symbol for i in range(1, 119)]

pairs = [
    f"{atomic_symbols[n]}-{atomic_symbols[m]}" 
    for n in range(118) 
    for m in range(n, 118) 
]

new_columns = ['id'] + pairs

df.columns = new_columns

df.to_csv('matriz_ligacoes.csv')