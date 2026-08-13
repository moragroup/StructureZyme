import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tempfile import TemporaryDirectory
from plip.structure.preparation import PDBComplex
from collections import Counter
from biotite.structure.io.pdb import PDBFile
from biotite.structure import AtomArrayStack

from structurezyme.steps.step import Step
from structurezyme.utils.helpers import SingleLigandSelect, suppress_stdout_stderr
from structurezyme.utils.helpers import get_hetatm_chain_ids, extract_chain_as_rdkit_mol, closest_ligands_by_element_composition
from structurezyme.utils.helpers import iter_substrates
from structurezyme.steps.geometric_filtering_cofactor_MCS import _suffix_keys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def select_ligand_from_smiles_via_composition(pdb_path: str | Path, substrate_smiles: str) -> str | None:
    """
    Choose the ligand whose element composition is closest to substrate_smiles,
    then return 'RESNUM:RESNAME:CHAIN' as input for fpocket. 
    """
    pdb_path = str(pdb_path)

    # find ligand chain ids
    chain_ids = get_hetatm_chain_ids(pdb_path)
    if not chain_ids:
        return None

    # get ligand mol objects
    chain_mols = []
    for ch in chain_ids:
        try:
            m = extract_chain_as_rdkit_mol(pdb_path, ch, sanitize=False)
        except Exception:
            m = None
        if m is not None:
            chain_mols.append((ch, m))

    if not chain_mols:
        return None

    # choose ligand closest by element composition
    mols_only = [m for _, m in chain_mols]
    ranked = closest_ligands_by_element_composition(mols_only, substrate_smiles, top_k=1)
    if not ranked:
        return None
    best_mol = ranked[0]

    # map back to chain id (first match)
    best_chain = next(ch for ch, m in chain_mols if m is best_mol)

    # read residue number and name for that chain 
    with open(pdb_path, "r") as f:
        pdb_file = PDBFile.read(f)
    structure = pdb_file.get_structure()
    if isinstance(structure, AtomArrayStack):
        structure = structure[0]

    mask = (structure.chain_id == best_chain)
    if not np.any(mask):
        return None

    res_ids = structure.res_id[mask] if hasattr(structure, "res_id") else structure.residue_id[mask]
    res_names = structure.res_name[mask] if hasattr(structure, "res_name") else structure.residue_name[mask]

    unique_ids = np.unique(res_ids)
    resnum = int(unique_ids.min())
    resseq  = int(np.min(res_ids))
    resname = Counter([str(r).strip() for r in res_names.tolist()]).most_common(1)[0][0]


    # return RESNUM:RESNAME:CHAIN (e.g., "1:LIG:B")
    return best_chain, resseq, resname



class PLIP(Step):
    def __init__(self, input_dir = None,  output_dir: str = ''):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)        

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.input_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            best_structure_name = row['docked_structure']
            substrate_smiles = row['substrate_smiles']
            pdb_file_as_path = Path(self.input_dir / f"{best_structure_name}.pdb")
            pdb_file_as_str = str(self.input_dir / f"{best_structure_name}.pdb")
            row_result = {}

            logger.info(f"Processing PDB file: {pdb_file_as_path.name}")

            if not pdb_file_as_path.exists():
                # Missing prepared PDB = upstream pipeline failure. Fail loudly
                # rather than swallow it into an empty (None) result row.
                raise FileNotFoundError(
                    f"Prepared PDB not found for {best_structure_name}: {pdb_file_as_path}"
                )

            # Default result structure
            default_result = {
                'plip_hydrogen_nbonds': None,
                'plip_hydrophobic_contacts': None,
                'plip_salt_bridges': None,
                'plip_pi_stacking': None,
                'plip_pi_cation': None,
                'plip_halogen_bonds': None,
                'plip_water_bridges': None,
                'plip_metal_complexes': None,
            }

            # load and analyze the docked structure (once per PDB). A load/analyze
            # failure is per-ROW (not per-substrate) so `prot` is left None and
            # every substrate falls back to a default (correctly-suffixed) block
            # below -- this never writes UNSUFFIXED keys in together mode.
            prot = None
            try:
                with suppress_stdout_stderr():
                    prot = PDBComplex()
                    prot.load_pdb(pdb_file_as_str)
                    prot.analyze()
            except Exception as e:
                logger.error(f"Error loading/analyzing PDB for {entry_name}: {e}")
                prot = None

            def _analyze(sub_smiles):
                # Per-substrate failures are isolated INSIDE the closure (as in
                # geometric_filtering_cofactor_MCS / ligandSASA) so that one bad
                # substrate yields its own suffixed None-block rather than
                # writing UNSUFFIXED keys that would mix with the good
                # substrate's `_s{i}` columns.
                r = dict(default_result)
                if prot is None:
                    return r
                try:
                    ligand = select_ligand_from_smiles_via_composition(
                        pdb_file_as_path, sub_smiles)
                    if not ligand:
                        return r
                    chain_id, resseq, resname = ligand
                    formatted_ligand_id = f"{resname}:{chain_id}:{resseq}"
                    interactions = prot.interaction_sets[formatted_ligand_id]
                    r['plip_hydrogen_nbonds'] = (
                        len(interactions.hbonds_ldon) + len(interactions.hbonds_pdon))
                    r['plip_hydrophobic_contacts'] = len(interactions.hydrophobic_contacts)
                    r['plip_salt_bridges'] = (
                        len(interactions.saltbridge_pneg) + len(interactions.saltbridge_lneg))
                    r['plip_pi_stacking'] = len(interactions.pistacking)
                    r['plip_pi_cation'] = (
                        len(interactions.pication_laro) + len(interactions.pication_paro))
                    r['plip_halogen_bonds'] = len(interactions.halogen_bonds)
                    r['plip_water_bridges'] = len(interactions.water_bridges)
                    r['plip_metal_complexes'] = len(interactions.metal_complexes)
                except Exception as e:
                    logger.error(f"Error processing {entry_name} substrate "
                                 f"{sub_smiles!r}: {e}")
                    return dict(default_result)
                return r

            subs = iter_substrates(row)
            if len(subs) <= 1:
                row_result.update(_analyze(substrate_smiles))
            else:
                for i, (s_smiles, _n, _m) in enumerate(subs):
                    row_result.update(_suffix_keys(_analyze(s_smiles), i))

            results.append(row_result)
        return results
                    

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            logger.warning("No output directory provided")
            return df

        results = self.__execute(df, self.output_dir)        
        results_df = pd.DataFrame(results) # Convert list of row-dictionaries to df       
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1) # Merge with input df

        return output_df
