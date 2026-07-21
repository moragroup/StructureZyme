from enzymetk.step import Step
import pandas as pd
from docko.docko import dock, get_alphafold_structure, clean_one_pdb, pdb_to_pdbqt_protein
import logging
import numpy as np
import os
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

    
class Vina(Step):
    
    def __init__(self, id_col: str, structure_col: str, sequence_col: str, 
                 substrate_col: str, substrate_name_col: str, active_site_col: str, output_dir: str, num_threads: int):
        logger.info('Expects active site residues as a string separated by |. Zero indexed.')
        self.id_col = id_col
        self.structure_col = structure_col
        self.sequence_col = sequence_col
        self.substrate_col = substrate_col
        self.substrate_name_col = substrate_name_col
        self.active_site_col = active_site_col  # Expects active site residues as a string separated by |
        self.output_dir = Path( output_dir) or None
        self.num_threads = num_threads or 1

    def _execute(self, df: pd.DataFrame) -> pd.DataFrame:
        output_filenames = []
        # ToDo: update to create from sequence if the path doesn't exist.
        for label, structure_path, seq, substrate_smiles, substrate_name, residues in df[[self.id_col, self.structure_col, self.sequence_col, self.substrate_col, self.substrate_name_col, self.active_site_col]].values:

            try:
                structure_path = str(structure_path)
                residues = [int(r) + 1 for r in str(residues).split('|') if r.strip()]
                if not residues:
                    logger.warning(f"Row {label}: empty active-site list, skipping")
                    output_filenames.append(None)
                    continue

                label_dir = self.output_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                structure_path = Path(structure_path)

                if not structure_path.exists():
                    # Try to download AF2 structure
                    get_alphafold_structure(label, label_dir / f"{label}_AF2.pdb")
                    structure_path = label_dir / f"{label}_AF2.pdb"

                # Skip if still not found
                if not structure_path.exists():
                    logger.warning(f"Skipping {label}: AF2 structure not found.")
                    output_filenames.append(None)
                    continue
            
                # Proceed with docking
                pdb_path = label_dir / f"{label}.pdb"
                pdbqt_path = label_dir / f"{label}.pdbqt"

                clean_one_pdb(str(structure_path), str(pdb_path))
                pdb_to_pdbqt_protein(str(pdb_path), str(pdbqt_path))

                score = dock(
                    sequence='',
                    protein_name=label,
                    smiles=substrate_smiles,
                    ligand_name=substrate_name,
                    residues=residues,
                    protein_dir=str(self.output_dir),
                    ligand_dir=str(self.output_dir),
                    output_dir=str(label_dir),
                    pH=7.4,
                    method='vina',
                    size_x=10.0,
                    size_y=10.0,
                    size_z=10.0
                )

                output_filenames.append(str(pdb_path))
                
            except Exception as e:
                logger.error(f'Error docking {label}: {e}')
                output_filenames.append(None)
            
        return output_filenames

 
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                pool = ThreadPool(self.num_threads)
                df_list = np.array_split(df, self.num_threads)
                results = pool.map(self._execute, df_list)
                pool.close()
                pool.join()
                # Flatten list of lists returned by pool.map
                results = [item for sublist in results for item in sublist]
            else:
                results = self._execute(df)
            df['output_dir'] = results
            return df
        else:
            logger.warning('No output directory provided')
