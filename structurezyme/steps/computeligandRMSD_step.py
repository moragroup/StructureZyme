import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os 
import re
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem, DataStructs
from rdkit.Geometry import Point3D
from Bio import PDB
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from scipy.spatial.distance import cdist  
from openbabel import openbabel as ob
from openbabel import pybel

from structurezyme.steps.step import Step
from structurezyme.utils.helpers import (
    clean_plt,
    get_hetatm_chain_ids,
    extract_chain_as_rdkit_mol,
    closest_ligands_by_element_composition,
    norm_l1_dist,
    atom_composition_fingerprint,
    _pose_id_from_structure_name,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global plot style
plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text
plt.rcParams['figure.figsize'] = (3,3)
sns.set(rc={'figure.figsize': (3,3), 'font.family': 'sans-serif', 'font.sans-serif': 'DejaVu Sans', 'font.size': 12}, 
        style='ticks')


def visualize_rmsd_by_entry(rmsd_df, output_dir="ligandRMSD_heatmaps"):
    '''
    Visualizes RMSD values as heatmaps for each entry in the resulting dataframe.
    '''   
    os.makedirs(output_dir, exist_ok=True)

    for entry, group in rmsd_df.groupby('Entry'):
        # Get all docked structures for the entry
        docked_proteins = list(set(group['docked_structure1']) | set(group['docked_structure2']))
        docked_proteins = sorted(docked_proteins, key=lambda x: (0 if "chai" in x.lower() else 1, x))
    
        rmsd_matrix = pd.DataFrame(np.nan, index=docked_proteins, columns=docked_proteins)

        for _, row in group.iterrows():
            l1, l2, rmsd = row['docked_structure1'], row['docked_structure2'], row['ligand_rmsd']
            rmsd_matrix.loc[l1, l2] = rmsd
            rmsd_matrix.loc[l2, l1] = rmsd

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(rmsd_matrix,annot=False, cmap='viridis', square=True, cbar=True)
        ax = clean_plt(ax)
        ax.set_title(f"Ligand RMSD Heatmap: {entry}", fontsize=14)
        ax.set_xlabel("Docked Structures")
        ax.set_ylabel("Docked Structures")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()
        filename = f"{entry.replace('/', '_')}_heatmap.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close() 


def get_tool_from_structure_name(structure_name: str) -> str:
    """
    Extracts the docking tool name from a structure string (e.g., 'Q97WW0_1_vina' -> 'vina').
    Assumes the tool is the last segment after the last underscore.

    fastrelax appends a '_relaxed' suffix to pose filenames
    (fastrelax_step.py:266), so 'P41365_0_chai_relaxed' must still resolve to
    'chai'. Strip a trailing '_relaxed' before taking the last token.
    """
    if structure_name.endswith('_relaxed'):
        structure_name = structure_name[:-len('_relaxed')]
    if '_' in structure_name:
        return structure_name.split('_')[-1]
    return "UNKNOWN_tool" # Fallback if format doesn't match


def pose_energy(docked_structure: str, entry_scores):
    """Return the fastrelax_score for a docked pose, or None if unavailable.

    `entry_scores` is one entry's per-engine dict:
    {"chai": {pose_key: score}, "boltz": {...}, "vina": {...}}.
    chai/boltz keys are `<Entry>_<pose_id>` (via _pose_id_from_structure_name);
    vina keys are the integer pose index int(stem.split('_')[-2]).
    """
    if not entry_scores:
        return None
    engine = get_tool_from_structure_name(docked_structure)
    scores = entry_scores.get(engine)
    if not scores:
        return None
    if engine == "vina":
        stem = docked_structure
        if stem.endswith("_relaxed"):
            stem = stem[: -len("_relaxed")]
        try:
            key = int(stem.split("_")[-2])
        except (ValueError, IndexError):
            return None
    else:
        key = _pose_id_from_structure_name(docked_structure)
    return scores.get(key)


def compute_normalized_ligand_rmsd_stats(rmsd_df: pd.DataFrame):
    """
    Computes per-entry normalized ligand RMSD statistics:
    - Mean and std of within-tool RMSDs (e.g. vina-vina, dchai-chai)
    - Mean and std of between-tool RMSDs (e.g. vina-chai, vina-boltz)
    - Overall mean and std RMSD for each entry
    Adds these as new columns to rmsd_df.
    """
    from itertools import combinations_with_replacement

    # Degrade gracefully when every pose pair failed RDKit processing
    # upstream (e.g. valence errors on corrupted ligands). Without this
    # guard, indexing `rmsd_df["tool1"]` on a 0-column DataFrame raises
    # KeyError and takes down the whole pipeline.
    if rmsd_df.empty or "tool1" not in rmsd_df.columns:
        logger.warning(
            "compute_normalized_ligand_rmsd_stats received an empty "
            "DataFrame (all pose pairs likely failed upstream). "
            "Returning an empty stats DataFrame."
        )
        return rmsd_df

    # Ensure lowercase and clean tool names
    rmsd_df["tool1"] = rmsd_df["tool1"].str.strip().str.lower()
    rmsd_df["tool2"] = rmsd_df["tool2"].str.strip().str.lower()

    enriched_df = rmsd_df.copy()

    # Collect per-entry statistics
    entry_stats = []

    for entry, group in rmsd_df.groupby("Entry"):
        stats = {"Entry": entry}
        overall_rmsds = []

        tools = sorted(set(group["tool1"]).union(group["tool2"]))

        for t1, t2 in combinations_with_replacement(tools, 2):
            pair_label = f"{min(t1, t2)}-{max(t1, t2)}"
            mask = group.apply(
                lambda row: set([row["tool1"], row["tool2"]]) == set([t1, t2]),
                axis=1
            )
            subset = group[mask]
            rmsds = subset["ligand_rmsd"].dropna().tolist()
            overall_rmsds.extend(rmsds)

            if t1 == t2:
                stats[f"{t1}_{t1}_mean_ligandRMSD"] = np.mean(rmsds) if rmsds else np.nan
                stats[f"{t1}_{t1}_std_ligandRMSD"] = np.std(rmsds) if rmsds else np.nan
            else:
                stats[f"{t1}_{t2}_mean_ligandRMSD"] = np.mean(rmsds) if rmsds else np.nan
                stats[f"{t1}_{t2}_std_ligandRMSD"] = np.std(rmsds) if rmsds else np.nan

        stats["overall_ligandRMSD_mean"] = np.mean(overall_rmsds) if overall_rmsds else np.nan
        stats["overall_ligandRMSD_std"] = np.std(overall_rmsds) if overall_rmsds else np.nan

        entry_stats.append(stats)

    stats_df = pd.DataFrame(entry_stats)
    enriched_df = enriched_df.merge(stats_df, on="Entry", how="left")

    return enriched_df


def _dense_rank(score_by_key: dict) -> dict:
    """Ascending dense rank (1 = smallest). Ties share a rank.

    score_by_key: {key: numeric}. Returns {key: rank_int}.
    """
    order = sorted(set(score_by_key.values()))
    rank_of_value = {v: i + 1 for i, v in enumerate(order)}
    return {k: rank_of_value[v] for k, v in score_by_key.items()}


def select_best_docked_structures(
    rmsd_df: pd.DataFrame,
    fastrelax_score_by_entry: dict | None = None,
) -> pd.DataFrame:
    """
    Selects the best overall docked structure per Entry using two methods:
    
    1. inter_tool_weighted_avg: Weighted average RMSD to all structures from other tools,
       weighted by number of structures each tool contributes.
       
    2. inter_tool_min_per_tool: Average of *minimum* RMSDs to each other tool
       (one closest structure per tool).

    3. vina_avg_intra_tool: Best structure among all vina generated structures based on
       lowest average RMSD to other Vina poses.

    4. fused_rank: energy-aware fusion of inter_tool_min_per_tool geometry with
       fastrelax_score (lower energy = better). Degrades to pure geometry when
       energy is unavailable.

    Returns a DataFrame with one row per method per Entry.
    """
    best_structures = []

    for entry, entry_df in rmsd_df.groupby("Entry"):
        # Build structure -> tool mapping
        structure_to_tool = {}
        all_structures = pd.unique(entry_df[['docked_structure1', 'docked_structure2']].values.ravel('K'))

        for _, row in entry_df.iterrows():
            structure_to_tool[row['docked_structure1']] = row['tool1']
            structure_to_tool[row['docked_structure2']] = row['tool2']

        tools = set(structure_to_tool.values())

        # Build RMSD matrix
        rmsd_matrix = pd.DataFrame(np.nan, index=all_structures, columns=all_structures)
        for _, row in entry_df.iterrows():
            s1, s2 = row['docked_structure1'], row['docked_structure2']
            rmsd_matrix.loc[s1, s2] = row['ligand_rmsd']
            rmsd_matrix.loc[s2, s1] = row['ligand_rmsd']
        np.fill_diagonal(rmsd_matrix.values, 0)

        # Group structures by tool
        tool_to_structures = defaultdict(list)
        for s, t in structure_to_tool.items():
            tool_to_structures[t].append(s)

        # ----- Method 1: Weighted average RMSD to all poses per other tool -----
        weighted_rmsd_scores = {}
        for s in rmsd_matrix.index:
            tool_s = structure_to_tool[s]
            weighted_sum = 0.0
            total_weight = 0

            for other_tool, other_structures in tool_to_structures.items():
                if other_tool == tool_s:
                    continue

                values = rmsd_matrix.loc[s, other_structures].dropna()
                if values.empty:
                    continue

                weight = len(other_structures)
                avg_rmsd = values.mean()

                weighted_sum += weight * avg_rmsd
                total_weight += weight

            if total_weight > 0:
                weighted_rmsd_scores[s] = weighted_sum / total_weight

        if weighted_rmsd_scores:
            best_s = min(weighted_rmsd_scores, key=weighted_rmsd_scores.get)
            best_structures.append({
                'Entry': entry,
                'tool': structure_to_tool[best_s],
                'best_structure': best_s,
                'avg_ligandRMSD': weighted_rmsd_scores[best_s],
                'method': 'inter_tool_weighted_avg'
            })

        # ----- Method 2: Average of closest pose per tool -----
        closest_rmsd_scores = {}
        for s in rmsd_matrix.index:
            tool_s = structure_to_tool[s]
            closest_sum = 0.0
            tool_count = 0

            for other_tool, other_structures in tool_to_structures.items():
                if other_tool == tool_s:
                    continue

                values = rmsd_matrix.loc[s, other_structures].dropna()
                if values.empty:
                    continue

                closest_sum += values.min()
                tool_count += 1

            if tool_count > 0:
                closest_rmsd_scores[s] = closest_sum / tool_count

        if closest_rmsd_scores:
            best_s = min(closest_rmsd_scores, key=closest_rmsd_scores.get)
            best_structures.append({
                'Entry': entry,
                'tool': structure_to_tool[best_s],
                'best_structure': best_s,
                'avg_ligandRMSD': closest_rmsd_scores[best_s],
                'method': 'inter_tool_min_per_tool'
            })

        # ----- Method 3: Intra-tool average RMSD within Vina structures -----
        vina_structures = tool_to_structures.get('vina', [])
        if len(vina_structures) >= 2:
            vina_matrix = rmsd_matrix.loc[vina_structures, vina_structures]
            avg_rmsd_vina = vina_matrix.mean(axis=1).dropna()

            if not avg_rmsd_vina.empty:
                best_vina = avg_rmsd_vina.idxmin()
                best_structures.append({
                    'Entry': entry,
                    'tool': 'vina',
                    'best_structure': best_vina,
                    'avg_ligandRMSD': avg_rmsd_vina[best_vina],
                    'method': 'vina_avg_intra_tool'
                })

        # ----- Method 4: energy-aware rank fusion (fused_rank) -----
        entry_scores = (
            (fastrelax_score_by_entry or {}).get(entry)
            if fastrelax_score_by_entry else None
        )

        # energy per structure (None where unrelaxed / unavailable)
        energy_by_s = {
            s: pose_energy(s, entry_scores) for s in rmsd_matrix.index
        }
        have_energy = {s: e for s, e in energy_by_s.items() if e is not None}

        # geometry score = inter_tool_min_per_tool value (lower = better)
        geom = dict(closest_rmsd_scores)  # may be empty for single-tool entries

        # degeneracy: <2 tools, or exactly 2 tools with a single-pose minority
        pose_counts = sorted(len(v) for v in tool_to_structures.values())
        degenerate_geometry = (
            len(tool_to_structures) < 2
            or (len(tool_to_structures) == 2 and pose_counts[0] == 1)
        )

        fused_best = None
        if len(have_energy) == 0:
            # No energy anywhere -> pure geometry (min_per_tool winner).
            if geom:
                fused_best = min(geom, key=geom.get)
        elif degenerate_geometry:
            # Geometry can't discriminate -> energy dominates, geometry tiebreak.
            fused_best = min(
                have_energy,
                key=lambda s: (have_energy[s], geom.get(s, float("inf")), s),
            )
        elif len(have_energy) >= 2 and geom:
            # Normal fusion: summed dense ranks; missing energy = worst rank.
            geom_rank = _dense_rank(geom)
            energy_rank = _dense_rank(have_energy)
            worst = (max(energy_rank.values()) if energy_rank else 0) + 1
            candidates = [s for s in geom_rank]  # all structures with geometry
            fused_best = min(
                candidates,
                key=lambda s: (
                    geom_rank[s] + energy_rank.get(s, worst),
                    energy_by_s[s] if energy_by_s[s] is not None else float("inf"),
                    s,
                ),
            )
        else:
            # 0 or 1 relaxed poses but geometry usable -> pure geometry.
            if geom:
                fused_best = min(geom, key=geom.get)

        if fused_best is not None:
            best_structures.append({
                'Entry': entry,
                'tool': structure_to_tool[fused_best],
                'best_structure': fused_best,
                'avg_ligandRMSD': geom.get(fused_best, float('nan')),
                'method': 'fused_rank',
            })

    return pd.DataFrame(best_structures)


class LigandRMSD(Step):
    def __init__(self, entry_col = 'Entry', input_dir: str = '', output_dir: str = '', visualize_heatmaps = False, maxMatches = 1000): 
        self.entry_col = entry_col
        self.input_dir = Path(input_dir)   
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_heatmaps = visualize_heatmaps
        self.maxMatches = maxMatches

    def __execute(self, df) -> list:

        rmsd_values = []

        # Iterate through all subdirectories in the input directory
        for sub_dir in self.input_dir.iterdir():
            logger.info(f"Processing entry: {sub_dir.name}")

            # Get substrate_smiles for entry
            try:
                substrate_smiles = df.loc[df[self.entry_col] == sub_dir.name, "substrate_smiles"].iloc[0]
                if pd.isna(substrate_smiles) or str(substrate_smiles).strip() == "":
                    logger.warning(f"[SKIP] substrate_smiles empty for {sub_dir.name}")
                    continue
            except IndexError:
                logger.warning(f"[SKIP] No substrate_smiles found for {sub_dir.name}")
                continue

            # Process all PDB files in subdirectories
            for pdb_file_path in sub_dir.glob("*.pdb"):

                # Extract chain IDs of ligands
                chain_ids = get_hetatm_chain_ids(pdb_file_path)

                # Extract ligands as RDKit mol objects
                ligands = []
                for chain_id in chain_ids:
                    mol  = extract_chain_as_rdkit_mol(pdb_file_path, chain_id, sanitize=False)
                    ligands.append(mol)

                filtered_ligands = closest_ligands_by_element_composition(ligands, substrate_smiles)

                if len(filtered_ligands) > 2:
                    logger.warning('More than 2 ligands were found matching the smile string.')
                    continue

                if len(filtered_ligands) == 0:
                    logger.warning(f"No valid ligands found for entry {sub_dir.name}. Skipping.")
                    continue

                ligand1 = filtered_ligands[0]
                ligand2 = filtered_ligands[1]

                if ligand1 is None or ligand2 is None:
                    logger.warning(f"Could not extract both ligands, skipping {pdb_file_path}")
                    continue

                try:
                    Chem.SanitizeMol(ligand1)
                    Chem.SanitizeMol(ligand2)
                    ligand1 = Chem.RemoveHs(ligand1)
                    ligand2 = Chem.RemoveHs(ligand2)

                    if ligand1.GetNumConformers() == 0:
                        AllChem.EmbedMolecule(ligand1)

                    if ligand2.GetNumConformers() == 0:
                        AllChem.EmbedMolecule(ligand2)

                except Chem.rdchem.AtomValenceException as e:
                    logger.warning(f"Valence error in {pdb_file_path.name}: {e}")
                    logger.debug(f"ligand1 SMILES: {Chem.MolToSmiles(ligand1)}")
                    logger.debug(f"ligand2 SMILES: {Chem.MolToSmiles(ligand2)}")
                    continue  # skip this ligand pair
                except Exception as e:
                    logger.error(f"Unexpected RDKit error in {pdb_file_path.name}: {e}")
                    continue

                # Calculate ligandRMSD
                try:
                    rmsd = rdMolAlign.CalcRMS(ligand1, ligand2, maxMatches=self.maxMatches)
                except RuntimeError as e:
                    logger.warning(f"LigandRMSD calculation failed for {pdb_file_path.name}: {e}")
                    continue 

                # Store the RMSD value in a dictionary
                pdb_file_name = pdb_file_path.name
                structure_names = pdb_file_name.replace(".pdb", "").split("__")
                entry_name = sub_dir.name 
                
                docked_structure1_name = structure_names[0] if len(structure_names) > 0 else None
                docked_structure2_name = structure_names[1] if len(structure_names) > 1 else None

                tool1_name = get_tool_from_structure_name(docked_structure1_name)
                tool2_name  = get_tool_from_structure_name(docked_structure2_name)

                rmsd_values.append({
                    'Entry': entry_name, 
                    'pdb_file': pdb_file_path.name,  # Store the name of the PDB file
                    'docked_structure1' : docked_structure1_name, 
                    'docked_structure2' : docked_structure2_name, 
                    'tool1' : tool1_name, 
                    'tool2': tool2_name,
                    'ligand_rmsd': rmsd   # Store the calculated RMSD value
                })

        # Pairwise ligandRMSD table
        rmsd_df = pd.DataFrame(rmsd_values)

        # Add tool-wise and overall normalized stats per entry
        rmsd_df = compute_normalized_ligand_rmsd_stats(rmsd_df)

        # If heatmaps are to be visualized, call the visualization function
        if self.visualize_heatmaps: 
            os.makedirs(Path(self.output_dir), exist_ok=True)
            visualize_rmsd_by_entry(rmsd_df, output_dir=Path(self.output_dir))

        # Select the best docked structures based on RMSD
        try:
            best_docked_structure_df = select_best_docked_structures(rmsd_df)

            # Merge metadata into pairwise df (keep pairwise as left table)
            rmsd_df = rmsd_df.merge(df, on='Entry', how='left')

            # Map each structure to the methods it was picked by
            best_map = (
                best_docked_structure_df
                .groupby(['Entry', 'best_structure'])['method']
                .apply(lambda x: ','.join(sorted(set(x))))
                .reset_index()
                .rename(columns={'best_structure': 'docked_structure', 'method': 'best_method'})
            )
            #print(best_map)
            # Per-structure single-row DataFrame
            per_entry_structures = pd.concat([
                rmsd_df[['Entry', 'docked_structure1']].rename(columns={'docked_structure1': 'docked_structure'}),
                rmsd_df[['Entry', 'docked_structure2']].rename(columns={'docked_structure2': 'docked_structure'})
            ], ignore_index=True).dropna(subset=['docked_structure']).drop_duplicates()

            # Get tool for each structure
            per_entry_structures['tool'] = per_entry_structures['docked_structure'].apply(get_tool_from_structure_name)

            # Merge stats per entry
            structures_df = per_entry_structures.merge(df, on='Entry', how='left', suffixes=('_drop', ''))
            # Drop the duplicates from per_entry_structures
            drop_cols = [c for c in structures_df.columns if c.endswith('_drop')]
            structures_df = structures_df.drop(columns=drop_cols)

            # Attach which selection methods (if any) chose this structure
            structures_df = structures_df.merge(
                best_map, on=['Entry', 'docked_structure'], how='left', validate='many_to_one'
            )

            # Flag sturctures deemed "best"
            structures_df['is_best'] = structures_df['best_method'].notna()
                    
            # aggregate to one row per (Entry, docked_structure)
            key_cols = ['Entry', 'docked_structure']
            agg_cols = [c for c in structures_df.columns if c not in key_cols]

            agg = {c: 'first' for c in agg_cols}  # default: take first value
            if 'best_method' in structures_df.columns:
                agg['best_method'] = lambda s: ','.join(sorted(set(s.dropna())))

            structures_df = (
                structures_df
                .groupby(key_cols, as_index=False)
                .agg(agg)
                .reset_index(drop=True)
            )

            # Determine if it's first structure generated by tool
            structures_df['is_first'] = structures_df['docked_structure'].apply(
                lambda s: s.split('_')[-2] == '0')
            

            # collect all *_mean_ligandRMSD / *_std_ligandRMSD columns that actually exist
            stat_cols_found = [c for c in rmsd_df.columns if re.search(r'_(mean|std)_ligandRMSD$', c)]
            overall_cols = [c for c in ["overall_ligandRMSD_mean", "overall_ligandRMSD_std"] if c in rmsd_df.columns]

            if stat_cols_found or overall_cols:
                entry_stats = rmsd_df[["Entry"] + stat_cols_found + overall_cols].drop_duplicates("Entry")
                structures_df = structures_df.merge(entry_stats, on="Entry", how="left")
            
            return rmsd_df, structures_df    
        except Exception as e:
            logger.error(f"Error selecting best docked structures: {e}")
            return rmsd_df, pd.DataFrame()  # Return empty DataFrame on error


    def execute(self, df) -> pd.DataFrame:
        self.input_dir = Path(self.input_dir)
        return self.__execute(df)
