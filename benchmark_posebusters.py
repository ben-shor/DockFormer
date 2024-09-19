import json
import os
import sys
from typing import Optional

import Bio.PDB
import Bio.SeqUtils
import numpy as np
import rdkit.Chem.rdMolAlign
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D

from run_pretrained_model import run_on_folder
from env_consts import POSEBUSTERS_JSONS, POSEBUSTERS_OUTPUT, POSEBUSTERS_GT, CKPT_PATH


def get_pdb_model(pdb_path: str):
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure("original_pdb", pdb_path)
    assert len(list(pdb_struct)) == 1, f"Too many models! {pdb_path}"
    return next(iter(pdb_struct))


def create_embeded_molecule(ref_mol: Chem.Mol, smiles: str):
    # Convert SMILES to a molecule
    target_mol = Chem.MolFromSmiles(smiles)

    assert target_mol is not None, f"Failed to parse molecule from SMILES {smiles}"

    # Set up parameters for conformer generation
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # Use all available threads
    params.pruneRmsThresh = 0.1  # Pruning threshold for RMSD

    # Generate multiple conformers
    num_conformers = 1000  # Define the number of conformers to generate
    conformer_ids = AllChem.EmbedMultipleConfs(target_mol, numConfs=num_conformers, params=params)

    # Optional: Optimize each conformer using MMFF94 force field
    # for conf_id in conformer_ids:
    #     AllChem.UFFOptimizeMolecule(target_mol, confId=conf_id)

    # Align each generated conformer to the initial aligned conformer of the target molecule
    rmsd_list = []
    for conf_id in conformer_ids:
        rmsd = rdMolAlign.AlignMol(target_mol, ref_mol, prbCid=conf_id)
        rmsd_list.append(rmsd)
        # print(f"Conformer ID: {conf_id}, RMSD: {rmsd:.4f}")

    # w = Chem.SDWriter(os.path.join('aligned_conformers.sdf'))
    # for conf_id in conformer_ids:
    #     w.write(target_mol, confId=conf_id)
    # w.close()

    best_rmsd_index = int(np.argmin(rmsd_list))
    # print(f"Best RMSD: {rmsd_list[best_rmsd_index]:.4f} (ind={best_rmsd_index})")
    # return target_mol.GetConformers()[best_rmsd_index]
    return target_mol, conformer_ids[best_rmsd_index]


def get_rmsd(gt_protein_path: str, gt_ligand_path: str, pred_protein_path: str, pred_ligand_path:str,
             reembed_smiles: Optional[str] = None, save_aligned: bool = False):

    gt_protein = get_pdb_model(gt_protein_path)
    pred_protein = get_pdb_model(pred_protein_path)

    assert len(list(gt_protein.get_chains())) == len(list(pred_protein.get_chains())), \
        "Different number of chains in proteins"
    chain_name_map = {a: b for a, b in zip(sorted([c.id for c in gt_protein.get_chains()]),
                                           sorted([c.id for c in pred_protein.get_chains()]))}

    gt_res_id_to_res, pred_res_id_to_res = {}, {}
    res_name_map = {}
    for (gt_chain_id, pred_chain_id) in chain_name_map.items():
        gt_res_in_chain = [res for res in gt_protein[gt_chain_id] if "CA" in res and
                           not (Bio.SeqUtils.seq1(res.get_resname()) in ("X", "", " "))]
        pred_res_in_chain = [res for res in pred_protein[pred_chain_id] if "CA" in res]
        assert len(gt_res_in_chain) == len(pred_res_in_chain), "Different number of residues in proteins"

        for gt_res, pred_res in zip(gt_res_in_chain, pred_res_in_chain):
            assert gt_res.get_resname() == pred_res.get_resname(), "Different residue names"
            gt_res_id = (gt_chain_id, str(gt_res.get_id()[1]) + str(gt_res.get_id()[2]))
            pred_res_id = (pred_chain_id, str(pred_res.get_id()[1]) + str(pred_res.get_id()[2]))
            gt_res_id_to_res[gt_res_id] = gt_res
            pred_res_id_to_res[pred_res_id] = pred_res
            res_name_map[gt_res_id] = pred_res_id

    super_imposer = Bio.PDB.Superimposer()
    ref_atoms = []
    sample_atoms = []
    for gt_res_id, pred_res_id in res_name_map.items():
        gt_res = gt_res_id_to_res[gt_res_id]
        pred_res = pred_res_id_to_res[pred_res_id]
        ref_atoms.append(gt_res["CA"])
        sample_atoms.append(pred_res["CA"])
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    protein_rmsd = super_imposer.rms

    # calculate pocket rmsd
    gt_protein_ca = [res for chain in gt_protein for res in chain]
    gt_ligand = Chem.SDMolSupplier(gt_ligand_path)[0]
    gt_ligand_atoms = [gt_ligand.GetConformer().GetAtomPosition(i) for i in range(gt_ligand.GetNumAtoms())]

    protein_ca_coords = np.array([i["CA"].get_coord() for i in gt_protein_ca])  # Shape: (n_protein, 3)
    ligand_atom_coords = np.array(gt_ligand_atoms)  # Shape: (n_ligand, 3)

    assert len(ref_atoms) == len(sample_atoms), f"Different number of CA atoms in proteins {len(ref_atoms)} {len(sample_atoms)}"
    # don't sanitize as when the positions are bad you have issues of "Can't kekulize mol."
    original_pred_ligand = Chem.MolFromMolFile(pred_ligand_path, sanitize=False)

    try:
        original_pred_ligand = Chem.RemoveHs(original_pred_ligand)
    except Exception as e:
        print("Failed to remove hydrogens", e)

    assert gt_ligand is not None, f"Failed to parse ligand from {gt_ligand_path}"
    assert original_pred_ligand is not None, f"Failed to parse ligand from {pred_ligand_path}"
    assert gt_ligand.GetNumAtoms() == original_pred_ligand.GetNumAtoms(), f"Different number of atoms in ligands " \
                                                                          f"{gt_ligand.GetNumAtoms()} {original_pred_ligand.GetNumAtoms()}"

    original_pred_ligand, conf_id = original_pred_ligand, 0
    if reembed_smiles:
        original_pred_ligand, conf_id = create_embeded_molecule(original_pred_ligand, reembed_smiles)
    pred_ligand = Chem.Mol(original_pred_ligand)

    # Compute squared distances between each protein CA and each ligand atom
    diff = protein_ca_coords[:, np.newaxis, :] - ligand_atom_coords[np.newaxis, :, :]  # Shape: (n_protein, n_ligand, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # Shape: (n_protein, n_ligand)
    is_interface = dist_sq <= 8 ** 2  # Boolean array of shape (n_protein, n_ligand)
    is_interface_res = np.any(is_interface, axis=1)  # Boolean array of shape (n_protein,)
    indices = np.where(is_interface_res)[0]

    gt_pocket_res_ids = []
    for i in indices:
        res = gt_protein_ca[i]
        gt_pocket_res_ids.append((res.parent.id, str(res.get_id()[1]) + str(res.get_id()[2])))

    pocket_most_common_chain = max(set([res_id[0] for res_id in gt_pocket_res_ids]),
                                     key=[res_id[0] for res_id in gt_pocket_res_ids].count)
    print(f"Number of interface residues: {len(indices)} {pocket_most_common_chain} {gt_pocket_res_ids}")

    super_imposer = Bio.PDB.Superimposer()
    ref_atoms = []
    sample_atoms = []
    for gt_res_id in gt_pocket_res_ids:
        if gt_res_id[0] != pocket_most_common_chain:
            continue
        pred_res_id = res_name_map[gt_res_id]
        gt_res = gt_res_id_to_res[gt_res_id]
        pred_res = pred_res_id_to_res[pred_res_id]
        ref_atoms.append(gt_res["CA"])
        sample_atoms.append(pred_res["CA"])
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    pocket_rmsd = super_imposer.rms

    # alt_chains = []
    # for possible_sample_chain_id in sorted([c.id for c in pred_protein.get_chains()]):
    #     if sample_chain_id == possible_sample_chain_id:
    #         continue


    # debug issues
    # for res1, res2 in zip(gt_protein[gt_chain_id].get_residues(), pred_protein.get_residues()):
    #     print(res1.get_id()[1], res1.get_resname(), res2.get_id()[1], res2.get_resname())

    # for a1, a2 in zip(ref_atoms, sample_atoms):
    #     res1, res2 = a1.get_parent(), a2.get_parent()
    #     print(res1.get_id()[1], res1.get_resname(), res2.get_id()[1], res2.get_resname())
    #
    # print([res.id[1] for res in gt_protein[gt_chain_id].get_residues() if "CA" in res])
    # print([res.id[1] for res in pred_protein.get_residues() if "CA" in res])

    pred_ligand = Chem.Mol(pred_ligand)
    pred_ligand_conf = pred_ligand.GetConformer(id=conf_id)
    # apply transformation to pred_ligand
    for i in range(pred_ligand.GetNumAtoms()):
        pos = pred_ligand_conf.GetAtomPosition(i)
        new_pos = np.dot(pos, super_imposer.rotran[0]) + super_imposer.rotran[1]
        pred_ligand_conf.SetAtomPosition(i, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
    # get the RMSD
    ligand_rmsd = rdkit.Chem.rdMolAlign.CalcRMS(pred_ligand, gt_ligand, prbId=conf_id)


    # check if there are alternative pockets
    original_pocket_res_ids = [k for k in gt_pocket_res_ids if k[0] == pocket_most_common_chain]
    for alt_gt_chain_id in chain_name_map.keys():
        if alt_gt_chain_id == pocket_most_common_chain:
            continue
        alt_pocket_res_ids = []
        for gt_res_id in gt_pocket_res_ids:
            alt_res_id = (alt_gt_chain_id, gt_res_id[1])
            if alt_res_id in gt_res_id_to_res:
                if gt_res_id_to_res[alt_res_id].get_resname() == gt_res_id_to_res[gt_res_id].get_resname():
                    alt_pocket_res_ids.append(alt_res_id)
        if len(alt_pocket_res_ids)/len(original_pocket_res_ids) > 0.7:
            alt_super_imposer = Bio.PDB.Superimposer()
            ref_atoms = []
            sample_atoms = []
            for gt_res_id in gt_pocket_res_ids:
                if gt_res_id[0] != pocket_most_common_chain:
                    continue
                alt_res_id = (alt_gt_chain_id, gt_res_id[1])
                if alt_res_id not in gt_res_id_to_res:
                    continue
                pred_res_id = res_name_map[gt_res_id]
                gt_res = gt_res_id_to_res[alt_res_id]
                pred_res = pred_res_id_to_res[pred_res_id]
                ref_atoms.append(gt_res["CA"])
                sample_atoms.append(pred_res["CA"])
            alt_super_imposer.set_atoms(ref_atoms, sample_atoms)

            alt_pred_ligand = Chem.Mol(original_pred_ligand)
            alt_pred_ligand_conf = alt_pred_ligand.GetConformer(id=conf_id)

            # apply transformation to pred_ligand
            for i in range(pred_ligand.GetNumAtoms()):
                pos = alt_pred_ligand_conf.GetAtomPosition(i)
                new_pos = np.dot(pos, alt_super_imposer.rotran[0]) + alt_super_imposer.rotran[1]
                alt_pred_ligand_conf.SetAtomPosition(i, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
            # get the RMSD
            alt_ligand_rmsd = rdkit.Chem.rdMolAlign.CalcRMS(alt_pred_ligand, gt_ligand, prbId=conf_id)
            print(f"Alternative pocket found {alt_gt_chain_id} {alt_super_imposer.rms} {alt_ligand_rmsd} ")

            if alt_ligand_rmsd < ligand_rmsd:
                print("Alternative is better", alt_ligand_rmsd, ligand_rmsd)
                ligand_rmsd = alt_ligand_rmsd
                pocket_rmsd = alt_super_imposer.rms
                super_imposer = alt_super_imposer
                pred_ligand = alt_pred_ligand

    # output aligned
    if save_aligned:
        aligned_pdb_path = pred_protein_path + "_aligned.pdb"
        io = Bio.PDB.PDBIO()
        super_imposer.apply(pred_protein.get_atoms())
        io.set_structure(pred_protein)
        io.save(aligned_pdb_path)
        aligned_ligand_path = pred_ligand_path + "_aligned.sdf"
        Chem.MolToMolFile(pred_ligand, aligned_ligand_path, confId=conf_id)

    return ligand_rmsd, pocket_rmsd, protein_rmsd


def simple_get_rmsd(protein_path1: str, protein_path2: str):
    gt_protein = get_pdb_model(protein_path1)
    pred_protein = get_pdb_model(protein_path2)

    super_imposer = Bio.PDB.Superimposer()
    ref_atoms = []
    sample_atoms = []
    for chain in gt_protein:
        for res in chain:
            if "CA" in res:
                ref_atoms.append(res["CA"])
    for chain in pred_protein:
        for res in chain:
            if "CA" in res:
                sample_atoms.append(res["CA"])
    assert len(ref_atoms) == len(sample_atoms), f"Different number of CA atoms in proteins {len(ref_atoms)} {len(sample_atoms)}"
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    return super_imposer.rms


VERSION2_IDS = ['5SAK_ZRY', '5SB2_1K2', '5SD5_HWI', '5SIS_JSM', '6M2B_EZO', '6M73_FNR', '6T88_MWQ', '6TW5_9M2', '6TW7_NZB', '6VTA_AKN', '6WTN_RXT', '6XBO_5MC', '6XCT_478', '6XG5_TOP', '6XHT_V2V', '6XM9_V55', '6YJA_2BA', '6YMS_OZH', '6YQV_8K2', '6YQW_82I', '6YR2_T1C', '6YRV_PJ8', '6YSP_PAL', '6YT6_PKE', '6YYO_Q1K', '6Z0R_Q4H', '6Z14_Q4Z', '6Z1C_7EY', '6Z2C_Q5E', '6Z4N_Q7B', '6ZAE_ACV', '6ZC3_JOR', '6ZCY_QF8', '6ZK5_IMH', '6ZPB_3D1', '7A1P_QW2', '7A9E_R4W', '7A9H_TPP', '7AFX_R9K', '7AKL_RK5', '7AN5_RDH', '7B2C_TP7', '7B94_ANP', '7BCP_GCO', '7BJJ_TVW', '7BKA_4JC', '7BMI_U4B', '7BNH_BEZ', '7BTT_F8R', '7C0U_FGO', '7C3U_AZG', '7C8Q_DSG', '7CD9_FVR', '7CIJ_G0C', '7CL8_TES', '7CNQ_G8X', '7CNS_PMV', '7CTM_BDP', '7CUO_PHB', '7D5C_GV6', '7D6O_MTE', '7DKT_GLF', '7DQL_4CL', '7DUA_HJ0', '7E4L_MDN', '7EBG_J0L', '7ECR_SIN', '7ED2_A3P', '7ELT_TYM', '7EPV_FDA', '7ES1_UDP', '7F51_BA7', '7F5D_EUO', '7F8T_FAD', '7FB7_8NF', '7FHA_ADX', '7FRX_O88', '7FT9_4MB', '7JG0_GAR', '7JHQ_VAJ', '7JMV_4NC', '7JXX_VP7', '7JY3_VUD', '7K0V_VQP', '7KB1_WBJ', '7KC5_BJZ', '7KM8_WPD', '7KQU_YOF', '7KRU_ATP', '7KZ9_XN7', '7L00_XCJ', '7L03_F9F', '7L5F_XNG', '7L7C_XQ1', '7LCU_XTA', '7LEV_0JO', '7LJN_GTP', '7LMO_NYO', '7LOE_Y84', '7LOU_IFM', '7LT0_ONJ', '7LZD_YHY', '7M31_TDR', '7M3H_YPV', '7M6K_YRJ', '7MFP_Z7P', '7MGT_ZD4', '7MGY_ZD1', '7MMH_ZJY', '7MOI_HPS', '7MSR_DCA', '7MWN_WI5', '7MWU_ZPM', '7MY1_IPE', '7MYU_ZR7', '7N03_ZRP', '7N4N_0BK', '7N4W_P4V', '7N6F_0I1', '7N7B_T3F', '7N7H_CTP', '7NF0_BYN', '7NF3_4LU', '7NFB_GEN', '7NGW_UAW', '7NLV_UJE', '7NP6_UK8', '7NPL_UKZ', '7NR8_UOE', '7NSW_HC4', '7NU0_DCL', '7NUT_GLP', '7NXO_UU8', '7O0N_CDP', '7O1T_5X8', '7ODY_DGI', '7OEO_V9Z', '7OFF_VCB', '7OFK_VCH', '7OLI_8HG', '7OMX_CNA', '7OP9_06K', '7OPG_06N', '7OSO_0V1', '7OZ9_NGK', '7OZC_G6S', '7P1F_KFN', '7P1M_4IU', '7P2I_MFU', '7P4C_5OV', '7P5T_5YG', '7PGX_FMN', '7PIH_7QW', '7PJQ_OWH', '7PK0_BYC', '7PL1_SFG', '7POM_7VZ', '7PRI_7TI', '7PRM_81I', '7PT3_3KK', '7PUV_84Z', '7Q25_8J9', '7Q27_8KC', '7Q2B_M6H', '7Q5I_I0F', '7QE4_NGA', '7QF4_RBF', '7QFM_AY3', '7QGP_DJ8', '7QHG_T3B', '7QHL_D5P', '7QPP_VDX', '7QTA_URI', '7R3D_APR', '7R59_I5F', '7R6J_2I7', '7R7R_AWJ', '7R9N_F97', '7RC3_SAH', '7RH3_59O', '7RKW_5TV', '7RNI_60I', '7ROR_69X', '7ROU_66I', '7RSV_7IQ', '7RWS_4UR', '7RZL_NPO', '7SCW_GSP', '7SDD_4IP', '7SFO_98L', '7SIU_9ID', '7SUC_COM', '7SZA_DUI', '7T0D_FPP', '7T1D_E7K', '7T3E_SLB', '7TB0_UD1', '7TBU_S3P', '7TE8_P0T', '7TH4_FFO', '7THI_PGA', '7TM6_GPJ', '7TOM_5AD', '7TS6_KMI', '7TSF_H4B', '7TUO_KL9', '7TXK_LW8', '7TYP_KUR', '7U0U_FK5', '7U3J_L6U', '7UAS_MBU', '7UAW_MF6', '7UJ4_OQ4', '7UJ5_DGL', '7UJF_R3V', '7ULC_56B', '7UMW_NAD', '7UQ3_O2U', '7USH_82V', '7UTW_NAI', '7UXS_OJC', '7UY4_SMI', '7UYB_OK0', '7V14_ORU', '7V3N_AKG', '7V3S_5I9', '7V43_C4O', '7VB8_STL', '7VBU_6I4', '7VC5_9SF', '7VKZ_NOJ', '7VQ9_ISY', '7VWF_K55', '7VYJ_CA0', '7W05_GMP', '7W06_ITN', '7WCF_ACP', '7WDT_NGS', '7WJB_BGC', '7WKL_CAQ', '7WL4_JFU', '7WPW_F15', '7WQQ_5Z6', '7WUX_6OI', '7WUY_76N', '7WY1_D0L', '7X5N_5M5', '7X9K_8OG', '7XBV_APC', '7XFA_D9J', '7XG5_PLP', '7XI7_4RI', '7XJN_NSD', '7XPO_UPG', '7XQZ_FPF', '7XRL_FWK', '7YZU_DO7', '7Z1Q_NIO', '7Z2O_IAJ', '7Z7F_IF3', '7ZCC_OGA', '7ZDY_6MJ', '7ZF0_DHR', '7ZHP_IQY', '7ZL5_IWE', '7ZOC_T8E', '7ZTL_BCN', '7ZU2_DHT', '7ZXV_45D', '7ZZW_KKW', '8A1H_DLZ', '8A2D_KXY', '8AAU_LH0', '8AEM_LVF', '8AIE_M7L', '8AP0_PRP', '8AQL_PLG', '8AUH_L9I', '8AY3_OE3', '8B8H_OJQ', '8BOM_QU6', '8BTI_RFO', '8C3N_ADP', '8C5M_MTA', '8CNH_V6U', '8CSD_C5P', '8D19_GSH', '8D39_QDB', '8D5D_5DK', '8DHG_T78', '8DKO_TFB', '8DP2_UMA', '8DSC_NCA', '8EAB_VN2', '8EX2_Q2Q', '8EXL_799', '8EYE_X4I', '8F4J_PHO', '8F8E_XJI', '8FAV_4Y5', '8FLV_ZB9', '8FO5_Y4U', '8G0V_YHT', '8G6P_API', '8GFD_ZHR', '8HFN_XGC', '8HO0_3ZI', '8SLG_G5A']

def main(config_path):
    use_relaxed = False
    use_reembed = False
    save_aligned = False
    assert not (use_relaxed and use_reembed), "Can't use both relaxed and reembed"

    i = 0
    while os.path.exists(os.path.join(POSEBUSTERS_OUTPUT, f"output_{i}")):
        i += 1
    output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_{i}")
    os.makedirs(output_dir, exist_ok=True)
    run_on_folder(POSEBUSTERS_JSONS, output_dir, config_path, long_sequence_inference=True)
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_10_run61")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_18_run85_67K")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_18_run85_67K")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_22_run87_93K")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_23_run86_260K")

    if not use_relaxed:
        jobnames = ["_".join(filename.split("_")[:2])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein.pdb") and "relaxed" not in filename]
    else:
        jobnames = ["_".join(filename.split("_")[:2])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein_relaxed.pdb")]
        # jobnames = ["_".join(filename.split("_")[1:3])
        #             for filename in os.listdir(os.path.join(output_dir, "predictions"))
        #             if filename.endswith("_protein.pdb") and "relaxed" in filename]

    print(f"Analyzing {len(jobnames)} jobs...")

    all_rmsds = {}
    for jobname in sorted(jobnames):
        gt_protein_path = os.path.join(POSEBUSTERS_GT, jobname, f"{jobname}_protein.pdb")
        gt_ligand_path = os.path.join(POSEBUSTERS_GT, jobname, f"{jobname}_ligand.sdf")
        pred_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_protein.pdb")
        pred_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_ligand_0.sdf")
        # relaxed_protein_path = os.path.join(output_dir, "predictions", f"relaxed_{jobname}_protein.pdb")
        # relaxed_ligand_path = os.path.join(output_dir, "predictions", f"relaxed_{jobname}_ligand.sdf")
        relaxed_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_protein_relaxed.pdb")
        relaxed_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_ligand_relaxed.sdf")

        input_json = os.path.join(POSEBUSTERS_JSONS, f"{jobname}.json")
        input_data = json.load(open(input_json, "r"))
        smiles = input_data["input_smiles"]

        parent_dir = os.path.dirname(POSEBUSTERS_JSONS)
        input_pdb_path = os.path.join(parent_dir, input_data["input_structure"])
        input_ref_sdf = os.path.join(parent_dir, input_data["ref_sdf"])

        if use_relaxed and not os.path.exists(relaxed_ligand_path):
            print("skipping, no relaxed", jobname)
            continue

        print(jobname)
        try:
            if use_relaxed:
                rmsds = get_rmsd(gt_protein_path, gt_ligand_path, relaxed_protein_path, relaxed_ligand_path,
                                 save_aligned=save_aligned)
            else:
                if not use_reembed:
                    smiles = None
                rmsds = get_rmsd(gt_protein_path, gt_ligand_path, pred_protein_path, pred_ligand_path,
                                 reembed_smiles=smiles, save_aligned=save_aligned)
        except Exception as e:
            print(f"Failed to compute RMSD for {jobname}", e)
            raise
            # raise e
            continue

        ligand_rmsd, pocket_rmsd, protein_rmsd = rmsds
        input_protein_to_pred_rmsd = simple_get_rmsd(input_pdb_path, pred_protein_path)
        input_protein_to_gt_rmsd = simple_get_rmsd(input_pdb_path, gt_protein_path)

        print(ligand_rmsd, pocket_rmsd, protein_rmsd, input_protein_to_pred_rmsd, input_protein_to_gt_rmsd)
        all_rmsds[jobname] = {"ligand_rmsd": ligand_rmsd, "pocket_rmsd": pocket_rmsd, "protein_rmsd": protein_rmsd,
                              "input_protein_to_pred_rmsd": input_protein_to_pred_rmsd,
                              "input_protein_to_gt_rmsd": input_protein_to_gt_rmsd}

    print(all_rmsds)
    json.dump(all_rmsds, open(os.path.join(output_dir, "rmsds.json"), "w"), indent=4)

    ligand_rmsds = {k: v["ligand_rmsd"] for k, v in all_rmsds.items()}
    print("Total: ", len(ligand_rmsds), "Under 2: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 2),
          "Under 5: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 5))

    print("-- version 2")
    ligand_rmsds = {k: v for k, v in ligand_rmsds.items() if k in VERSION2_IDS}
    print("Total: ", len(ligand_rmsds), "Under 2: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 2),
          "Under 5: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 5))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    main(config_path)
