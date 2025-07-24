# DockFormer

Variation of AlphaFold2 for flexible docking of ligands in proteins. The main alterations to the AlphaFold2 code:
* Integrate ligand input for predictions of protein-ligand interactions.
* Affinity head that predicts the affinity of the interaction.
* Use apo protein structure as input (can be predicted based on a target sequence using AF2), so no need to refold the protein for each ligand.
* Instead of 48 layers of EvoFormer, only 8 layers of a lightweight Pairformer (~3s per prediction)

Read the paper: https://www.biorxiv.org/content/10.1101/2024.11.25.625135v2

Based on the OpenFold repository (https://github.com/aqlaboratory/openfold).
