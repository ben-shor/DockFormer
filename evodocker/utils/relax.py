import sys
import os.path
import time
from random import shuffle

import numpy as np
import pdbfixer
import openmm as mm
import openmm.app as mm_app
import openmm.unit as mm_unit
from openmm import CustomExternalForce
from openmm.app import Modeller
from openmmforcefields.generators import SystemGenerator
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import UndefinedStereochemistryError, RadicalsNotSupportedError
import mdtraj
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import Bio.PDB
from Bio.SVDSuperimposer import SVDSuperimposer


# -- Relax protein and ligand. Code adapted from:
# https://github.com/patrickbryant1/Umol/blob/f7cd2b4de09b4e7cc1b68606791dd1cc81deeebc/src/relax/openmm_relax.py
def fix_pdb(pdb_path, hydrogen_added_pdb_path):
    """Add hydrogens to the PDB file
    """
    fixer = pdbfixer.PDBFixer(pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    mm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open(hydrogen_added_pdb_path, 'w'))
    return fixer.topology, fixer.positions


def minimize_energy(topology, system, positions, output_pdb_path):
    '''Function that minimizes energy, given topology, OpenMM system, and positions '''
    # Use a Brownian Integrator
    integrator = mm.BrownianIntegrator(
        100 * mm.unit.kelvin,
        100. / mm.unit.picoseconds,
        2.0 * mm.unit.femtoseconds
    )
    simulation = mm.app.Simulation(topology, system, integrator)

    # Initialize the DCDReporter
    reportInterval = 100  # Adjust this value as needed
    reporter = mdtraj.reporters.DCDReporter('positions.dcd', reportInterval)

    # Add the reporter to the simulation
    simulation.reporters.append(reporter)

    simulation.context.setPositions(positions)

    simulation.minimizeEnergy(1, 1000)
    # Save positions
    minpositions = simulation.context.getState(getPositions=True).getPositions()
    mm_app.PDBFile.writeFile(topology, minpositions, open(output_pdb_path, "w"))

    reporter.close()

    return topology, minpositions


def add_restraints(system, topology, positions, restraint_type):
    # Code adapted from https://gist.github.com/peastman/ad8cda653242d731d75e18c836b2a3a5
    restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    system.addForce(restraint)
    restraint.addGlobalParameter('k', 100.0*mm_unit.kilojoules_per_mole/mm_unit.nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atom in topology.atoms():
        if restraint_type == 'protein':
            if 'x' not in atom.name:
                restraint.addParticle(atom.index, positions[atom.index])
        elif restraint_type == 'CA+ligand':
            if ('x' in atom.name) or (atom.name == "CA"):
                restraint.addParticle(atom.index, positions[atom.index])

    return system


def create_joined_relaxed(protein_pdb_path: str, ligand_sdf_path: str, hydorgen_added_protein_pdb_path: str,
                          relaxed_joined_path: str):
    restraint_type = 'CA+ligand'

    start_time = time.time()
    print('Reading ligand')
    try:
        ligand_mol = Molecule.from_file(ligand_sdf_path)
    # Check for undefined stereochemistry, allow undefined stereochemistry to be loaded
    except UndefinedStereochemistryError:
        print('Undefined Stereochemistry Error found! Trying with undefined stereo flag True')
        ligand_mol = Molecule.from_file(ligand_sdf_path, allow_undefined_stereo=True)
    # Check for radicals -- break out of script if radical is encountered
    except RadicalsNotSupportedError:
        print('OpenFF does not currently support radicals -- use unrelaxed structure')
        sys.exit()
    # Assigning partial charges first because the default method (am1bcc) does not work
    ligand_mol.assign_partial_charges(partial_charge_method='gasteiger')

    # Read protein PDB and add hydrogens
    protein_topology, protein_positions = fix_pdb(protein_pdb_path, hydorgen_added_protein_pdb_path)
    print('Added all atoms...')

    modeller = Modeller(protein_topology, protein_positions)
    print('System has %d atoms' % modeller.topology.getNumAtoms())

    print('Adding ligand...')
    lig_top = ligand_mol.to_topology()
    modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
    print('System has %d atoms' % modeller.topology.getNumAtoms())

    print('Preparing system')
    # Initialize a SystemGenerator using the GAFF for the ligand and implicit water.
    # forcefield_kwargs = {'constraints': mm_app.HBonds, 'rigidWater': True, 'removeCMMotion': False,
    # 'hydrogenMass': 4*mm_unit.amu }
    system_generator = SystemGenerator(
        forcefields=['amber14-all.xml', 'implicit/gbn2.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=[ligand_mol],
        # forcefield_kwargs=forcefield_kwargs
    )

    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    print('Adding restraints on protein CAs and ligand atoms')

    system = add_restraints(system, modeller.topology, modeller.positions, restraint_type=restraint_type)

    minimize_energy(modeller.topology, system, modeller.positions, relaxed_joined_path)

    print(f'Time taken for relax calculation is {time.time() - start_time:.1f} seconds')


# -- Fix ligand changed structure. Code adapted from:
# https://github.com/patrickbryant1/Umol/blob/f7cd2b4de09b4e7cc1b68606791dd1cc81deeebc/src/relax/align_ligand_conformer.py
def generate_best_conformer(pred_coords, ligand_smiles, max_confs=100):
    """Generate conformers and compare the coords with the predicted atom positions

    Generating with constraints doesn't seem to work.
    cids = Chem.rdDistGeom.EmbedMultipleConfs(m,max_confs,ps)
    if len([x for x in m.GetConformers()])<1:
        print('Could not generate conformer with constraints')
    """
    # Generate conformers
    m = Chem.AddHs(Chem.MolFromSmiles(ligand_smiles))
    # Embed in 3D to get distance matrix
    AllChem.EmbedMolecule(m, maxAttempts=500)
    bounds = AllChem.Get3DDistanceMatrix(m)
    # Get pred distance matrix
    pred_dmat = np.sqrt(1e-10 + np.sum((pred_coords[:, None] - pred_coords[None, :]) ** 2 ,axis=-1))
    # Go through the atom types and add the constraints if not H
    # The order here will be the same as for the pred ligand as the smiles are identical
    ai, mi = 0, 0
    bounds_mapping = {}
    for atom in m.GetAtoms():
        if atom.GetSymbol() != 'H':
            bounds_mapping[ai] = mi
            ai += 1
        mi += 1

    # Assign available pred bound atoms
    bounds_keys = [*bounds_mapping.keys()]
    for i in range(len(bounds_keys)):
        key_i = bounds_keys[i]
        for j in range(i+1, len(bounds_keys)):
            key_j = bounds_keys[j]
            try:
                bounds[bounds_mapping[key_i], bounds_mapping[key_j]] = pred_dmat[i, j]
                bounds[bounds_mapping[key_j], bounds_mapping[key_i]] = pred_dmat[j, i]
            except:
                continue
    # Now generate conformers using the bounds
    ps = Chem.rdDistGeom.ETKDGv3()
    ps.randomSeed = 0xf00d
    ps.SetBoundsMat(bounds)
    cids = Chem.rdDistGeom.EmbedMultipleConfs(m, max_confs)
    # Get all conformer dmats
    nonH_inds = [*bounds_mapping.values()]
    conf_errs = []
    for conf in m.GetConformers():
        pos = conf.GetPositions()
        nonH_pos = pos[nonH_inds]
        conf_dmat = np.sqrt(1e-10 + np.sum((nonH_pos[:,None]-nonH_pos[None,:])**2,axis=-1))
        err = np.mean(np.sqrt(1e-10 + (conf_dmat-pred_dmat)**2))
        conf_errs.append(err)

    # Get the best
    best_conf_id = np.argmin(conf_errs)
    best_conf_err = conf_errs[best_conf_id]
    best_conf = [x for x in m.GetConformers()][best_conf_id]
    best_conf_pos = best_conf.GetPositions()

    return best_conf, best_conf_pos, best_conf_err, [atom.GetSymbol() for atom in m.GetAtoms()], nonH_inds, m, best_conf_id


def align_coords_transform(pred_pos, conf_pos, nonH_inds):
    """Align the predicted and conformer positions
    """
    sup = SVDSuperimposer()

    sup.set(pred_pos, conf_pos[nonH_inds])  # (reference_coords, coords)
    sup.run()
    rot, tran = sup.get_rotran()

    # Rotate coords from new chain to its new relative position/orientation
    tr_coords = np.dot(conf_pos, rot) + tran

    return tr_coords


def write_sdf(mol, conf, aligned_conf_pos, best_conf_id, outname):
    for i in range(mol.GetNumAtoms()):
        x, y, z = aligned_conf_pos[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    writer = Chem.SDWriter(outname)
    writer.write(mol, confId=int(best_conf_id))


# Main function
def relax_complex(protein_pdb_path: str, ligand_sdf_path: str, relaxed_protein_path: str, relaxed_ligand_path: str):
    hydorgen_added_protein_pdb_path = protein_pdb_path + "_hydrogen_added.pdb"
    relaxed_joined_path = protein_pdb_path + "_joined_relaxed.pdb"

    create_joined_relaxed(protein_pdb_path, ligand_sdf_path, hydorgen_added_protein_pdb_path, relaxed_joined_path)

    parser = Bio.PDB.PDBParser(QUIET=True)
    joined_structure = next(iter(parser.get_structure('', relaxed_joined_path)))

    # save the relaxed protein
    io = Bio.PDB.PDBIO()
    io.set_structure(joined_structure["A"])
    io.save(relaxed_protein_path)

    relaxed_ligand_coords = np.array([atom.get_coord() for atom in joined_structure["B"].get_atoms()
                                      if atom.get_id()[0] != "H"])
    original_ligand = Chem.SDMolSupplier(ligand_sdf_path)[0]
    ligand_smiles = Chem.MolToSmiles(original_ligand)

    best_conf, best_conf_pos, best_conf_err, atoms, nonH_inds, mol, best_conf_id = generate_best_conformer(
        relaxed_ligand_coords, ligand_smiles, max_confs=100
    )

    aligned_conf_pos = align_coords_transform(relaxed_ligand_coords, best_conf_pos, nonH_inds)

    write_sdf(mol, best_conf, aligned_conf_pos, best_conf_id, relaxed_ligand_path)


def relax_folder(folder_path: str):
    all_jobnames = []
    filenames = os.listdir(folder_path)
    shuffle(filenames)
    for filename in filenames:
        if filename.endswith("_predicted_protein.pdb"):
            jobname = filename.split("_predicted_protein.pdb")[0]
            ligand_path = os.path.join(folder_path, jobname + "_predicted_ligand_0.sdf")
            if not os.path.exists(ligand_path):
                continue
            all_jobnames.append(jobname)

    success = 0
    for jobname in all_jobnames:
        protein_pdb_path = os.path.join(folder_path, jobname + "_predicted_protein.pdb")
        ligand_sdf_path = os.path.join(folder_path, jobname + "_predicted_ligand_0.sdf")
        relaxed_protein_path = os.path.join(folder_path, jobname + "_protein_relaxed.pdb")
        relaxed_ligand_path = os.path.join(folder_path, jobname + "_ligand_relaxed.sdf")
        if os.path.exists(relaxed_protein_path) and os.path.exists(relaxed_ligand_path):
            print("Already has relaxed", jobname)
            success += 1
            continue
        print("Relaxing", jobname)
        try:
            relax_complex(protein_pdb_path, ligand_sdf_path, relaxed_protein_path, relaxed_ligand_path)
            success += 1
        except Exception as e:
            print("Failed to relax", jobname, e)
    print(f"Relaxed {success}/{len(all_jobnames)}")


if __name__ == "__main__":
    relax_folder(os.path.abspath(sys.argv[1]))
