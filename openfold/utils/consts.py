from rdkit import Chem

# Survey of atom types in the PDBBind-short
# [('C', 216310), ('O', 52539), ('N', 45005), ('S', 3706), ('F', 3081), ('Cl', 1847), ('P', 1805), ('Br', 332),
# ('B', 124), ('H', 111), ('I', 106), ('Se', 12), ('Fe', 8), ('Ru', 5), ('Ir', 4), ('Co', 3), ('Si', 3), ('Pt', 2),
# ('Rh', 1), ('Cu', 1), ('Re', 1), ('V', 1), ('Mg', 1)]
POSSIBLE_ATOM_TYPES = ["C", "O", "N", "S", "F", "Cl", "P", "Br", "B", "H", "I", "Se", "Fe", "Ru", "Ir", "other"]


POSSIBLE_BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                       Chem.rdchem.BondType.AROMATIC, "other"]