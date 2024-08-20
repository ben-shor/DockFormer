from rdkit.Chem.rdchem import ChiralType, BondType

# Survey of atom types in the PDBBind
# {'C': 403253, 'O': 101283, 'N': 81325, 'S': 6262, 'F': 5256, 'P': 3378, 'Cl': 2920, 'Br': 552, 'B': 237, 'I': 185,
#  'H': 181, 'Fe': 19, 'Se': 15, 'Ru': 10, 'Si': 5, 'Co': 4, 'Ir': 4, 'As': 2, 'Pt': 2, 'V': 1, 'Mg': 1, 'Be': 1,
#  'Rh': 1, 'Cu': 1, 'Re': 1}
POSSIBLE_ATOM_TYPES = ['C', 'O', 'N', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'I', 'H', 'Fe', 'Se', 'Ru', 'Si', 'Co', 'Ir',
                       'As', 'Pt', 'V', 'Mg', 'Be', 'Rh', 'Cu', 'Re']

# bonds Counter({BondType.SINGLE: 366857, BondType.AROMATIC: 214238, BondType.DOUBLE: 59725, BondType.TRIPLE: 866,
# BondType.UNSPECIFIED: 18, BondType.DATIVE: 8})
POSSIBLE_BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.UNSPECIFIED,
                       BondType.DATIVE]

# {0: 580061, 1: 13273, -1: 11473, 2: 44, 7: 17, -2: 8, 9: 7, 10: 7, 5: 3, 3: 3, 4: 1, 6: 1, 8: 1}
POSSIBLE_CHARGES = [-1, 0, 1]

# {ChiralType.CHI_UNSPECIFIED: 551374, ChiralType.CHI_TETRAHEDRAL_CCW: 27328, ChiralType.CHI_TETRAHEDRAL_CW: 26178,
# ChiralType.CHI_OCTAHEDRAL: 13, ChiralType.CHI_SQUAREPLANAR: 3, ChiralType.CHI_TRIGONALBIPYRAMIDAL: 3}
POSSIBLE_CHIRALITIES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_TETRAHEDRAL_CW,
                        ChiralType.CHI_OCTAHEDRAL, ChiralType.CHI_SQUAREPLANAR, ChiralType.CHI_TRIGONALBIPYRAMIDAL]

