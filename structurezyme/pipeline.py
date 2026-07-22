from pathlib import Path
from typing import Union
import pandas as pd
import logging
import os
import time
import psutil
from functools import wraps

from structurezyme.utils.helpers import log_section, log_subsection, log_boxed_note, generate_boltz_structure_path, generate_chai_structure_path
from structurezyme.utils.helpers import clean_protein_sequence, delete_empty_subdirs, extract_docking_metrics, valid_file_list, add_metrics
from structurezyme.utils.helpers import log_usage
from structurezyme.steps.save_step import Save
from structurezyme.steps.dock_vina_step import Vina
from structurezyme.steps.extract_docking_metrics_step import DockingMetrics
from structurezyme.steps.preparevina_step import PrepareVina
from structurezyme.steps.preparechai_step import PrepareChai
from structurezyme.steps.prepareboltz_step import PrepareBoltz
from structurezyme.steps.superimposestructures_step import SuperimposeStructures
from structurezyme.steps.computeproteinRMSD_step import ProteinRMSD
from structurezyme.steps.computeligandRMSD_step import LigandRMSD
from structurezyme.steps.geometric_filtering_cofactor_MCS import GeneralGeometricFiltering
from structurezyme.steps.geometric_filtering_esterase import EsteraseGeometricFiltering
from structurezyme.steps.fpocket_step import Fpocket
from structurezyme.steps.ligandSASA_step import LigandSASA
from structurezyme.steps.plip_step import PLIP
from structurezyme.steps.fastrelax_step import FastRelax

from enzymetk.dock_chai_step import Chai
from enzymetk.dock_boltz_step import Boltz
from structurezyme.steps.squidly_step import Squidly

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

"""
Pipeline v2 supports Chai + Boltz docking by default, with optional Vina docking.

For de novo designed enzymes, Vina docking can be skipped (run_vina=False) because
catalytic residue prediction may not be reliable. For non-de-novo enzymes with known
active sites, enable Vina docking (run_vina=True) for physics-based docking.
"""


class Docking:
    def __init__(
        self,
        df: pd.DataFrame,
        boltz_cache_dir: str,
        output_dir: Union[str, Path] = "pipeline_output",
        squidly_dir: Union[str, Path] = '',
        metagenomic_enzymes: int = 0,
        skip_catalytic_residue_prediction: bool = False,
        run_vina: bool = False,
        alternative_structure_for_vina: str = 'Chai',
        use_msa_server: bool = True,
        num_threads: int = 1,
        squidly_model_size: str = '3B',
        squidly_as_threshold: float | None = None,
        squidly_num_threads: int | None = None,
    ):
        self.df = df.copy()
        self.boltz_cache_dir = boltz_cache_dir
        self.squidly_dir = Path(squidly_dir) 
        self.skip_catalytic_residue_prediction = skip_catalytic_residue_prediction
        self.run_vina = run_vina
        self.alternative_structure_for_vina = alternative_structure_for_vina
        self.use_msa_server = use_msa_server
        self.num_threads = num_threads
        self.squidly_model_size = squidly_model_size
        self.squidly_as_threshold = squidly_as_threshold
        self.squidly_num_threads = squidly_num_threads if squidly_num_threads is not None else num_threads
        self.output_dir = Path(output_dir)
        self.metagenomic_enzymes = metagenomic_enzymes
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        
        if self.skip_catalytic_residue_prediction:
            log_section("Skipping catalytic residue prediction")
            df_squidly = self.df
            if self.run_vina:
                has_vina_residues = (
                    'vina_residues' in df_squidly.columns
                    and df_squidly['vina_residues'].astype(str).str.strip().ne('').any()
                )
                if not has_vina_residues:
                    log_boxed_note(
                        "WARNING: skip_catalytic_residue_prediction=True with run_vina=True "
                        "and no vina_residues provided. Vina will have no pocket definition "
                        "and entries without vina_residues will be dropped before docking."
                    )
        else:
            log_section("Predicting active site residues")
            df_squidly = self._catalytic_residue_prediction()

        log_section("Protein-Ligand docking")
        df_chai = self._run_chai(df_squidly)
        df_boltz = self._run_boltz(df_chai)

        if self.run_vina:
            df_docked = self._run_vina(df_boltz)
        else:
            df_docked = df_boltz

        df_metrics = self._extract_docking_quality_metrics(df_docked)

    @log_usage("Predict catalytic residues")
    def _catalytic_residue_prediction(self):
        self.df['Sequence'] = self.df['Sequence'].apply(clean_protein_sequence)

        # The Squidly wrapper handles: CLI invocation, sequence dedup,
        # column renaming (Squidly_Ensemble_Residues -> Squidly_CR_Position),
        # and broadcasting predictions back to the full DataFrame.
        squidly_step = Squidly(
            sequence_col='Sequence',
            id_col='Entry',
            model_size=self.squidly_model_size,
            as_threshold=self.squidly_as_threshold,
            num_threads=self.squidly_num_threads,
        )
        df_squidly = squidly_step.execute(self.df)

        # Remove entries without catalytic residues for proteins without user-specified residues for vina-docking
        if 'vina_residues' not in df_squidly.columns:
            df_squidly['vina_residues'] = None

        # Ensure both columns exist and normalize to clean strings
        for col in ['Squidly_CR_Position', 'vina_residues']:
            if col not in df_squidly.columns:
                df_squidly[col] = ''
            # Convert list-like to pipe-delimited; keep strings as-is; clean up None/NaN/whitespace
            def _norm(v):
                if v is None:
                    return ''
                if isinstance(v, (list, tuple)):
                    return '|'.join(str(x).strip() for x in v if str(x).strip() != '')
                s = str(v).strip()
                return '' if s.lower() in ('nan', 'none', '[]') else s
            df_squidly[col] = df_squidly[col].apply(_norm)

        # Remove entries with *no* catalytic residues from either source
        mask_empty = (df_squidly['Squidly_CR_Position'] == '') & (df_squidly['vina_residues'] == '')
        empty_entries = df_squidly.loc[mask_empty, 'Entry'].tolist()
        if empty_entries:
            log_boxed_note(
                'Removing entries without catalytic residues and without specified residues for vina docking: '
                + ', '.join(empty_entries)
            )
        df_squidly = df_squidly[~mask_empty].reset_index(drop=True)

        # Prefer user-specified vina residues when provided; else use Squidly
        use_vina = df_squidly['vina_residues'] != ''
        df_squidly['catalytic_residues'] = df_squidly['vina_residues'].where(use_vina, df_squidly['Squidly_CR_Position'])

        output_path = os.path.join(self.output_dir, 'squidly.pkl')
        df_squidly.to_pickle(output_path)
        log_boxed_note("Finished predicting active site residues")
        return df_squidly

    @log_usage("Running Chai docking")
    def _run_chai(self, df_squidly):
        log_subsection("Docking using Chai")
        chai_dir = Path(self.output_dir) / 'chai'
        chai_dir.mkdir(exist_ok=True, parents=True)
        if 'cofactor_smiles' not in df_squidly.columns:
            df_squidly['cofactor_smiles'] = ''
        df_chai = df_squidly << (Chai('Entry', 'Sequence', 'substrate_smiles', 'cofactor_smiles', chai_dir, self.num_threads) >> Save(Path(self.output_dir)/'chai.pkl'))
        df_chai.rename(columns = {'output_dir':'chai_dir'}, inplace=True)
        return df_chai

    @log_usage("Running Boltz docking")
    def _run_boltz(self, df_chai):
        log_subsection("Docking using Boltz")
        boltz_dir = Path(self.output_dir) / 'boltz/'
        boltz_dir.mkdir(exist_ok=True, parents=True)
        if 'cofactor_smiles' not in df_chai.columns:
            df_chai['cofactor_smiles'] = None

        # Build args dynamically from parameters
        boltz_args = ['--cache', str(self.boltz_cache_dir)]
        if self.use_msa_server:
            boltz_args.append('--use_msa_server')
        # `--no_kernels` disables Boltz's optional CUDA fused kernels that
        # require the `cuequivariance_ops_torch` package. Without this flag,
        # Boltz inference raises ImportError mid-forward, silently leaves an
        # empty `predictions/` directory, and the pipeline downstream sees no
        # CIF files (boltz_files_for_superimposition=[]). That in turn yields
        # an empty LigandRMSD output and a KeyError on `docked_structure` in
        # extract_docking_metrics. The pure-PyTorch fallback is slower but
        # always available.
        boltz_args.append('--no_kernels')

        df_boltz = df_chai << (Boltz('Entry', 'Sequence', 'substrate_smiles', 'cofactor_smiles', boltz_dir, self.num_threads, 
                                     args=boltz_args)
                            >> Save(Path(self.output_dir)/'boltz.pkl'))
        df_boltz.rename(columns = {'output_dir':'boltz_dir'}, inplace=True)
        return df_boltz

    @log_usage("Running Vina docking")
    def _run_vina(self, df_boltz):
        log_subsection("Docking using Vina")
        vina_dir = Path(self.output_dir) / 'vina/'
        vina_dir.mkdir(exist_ok=True, parents=True)
        delete_empty_subdirs(vina_dir)

        if self.metagenomic_enzymes == 1:
            if self.alternative_structure_for_vina == 'Chai':
                log_boxed_note('Fallback to Chai structures for docking due to missing AF2 structures.')    
                df_boltz['structure'] = df_boltz['chai_dir'].apply(generate_chai_structure_path)
            elif self.alternative_structure_for_vina == 'Boltz':
                log_boxed_note('Fallback to Boltz structures for docking due to missing AF2 structures.')    
                df_boltz['structure'] = df_boltz['boltz_dir'].apply(generate_boltz_structure_path)
        else: 
            df_boltz['structure'] = None  # or path to AF structure
        
        # Initial Vina docking attempt
        df_vina = df_boltz << (Vina('Entry', 'structure', 'Sequence', 'substrate_smiles', 'substrate_name', 'catalytic_residues', vina_dir, self.num_threads))
        df_vina.rename(columns = {'output_dir':'vina_dir'}, inplace=True)

        # Handle missing AF2 structures by retrying with alternative structure
        if df_vina['vina_dir'].isnull().any():
            missing_entries = df_vina[df_vina['vina_dir'].isnull()]['Entry'].unique()
            delete_empty_subdirs(vina_dir)    

            if self.alternative_structure_for_vina == 'Chai':
                log_boxed_note('Fallback to Chai structures for docking due to missing AF2 structures. ' + f'Entries: {list(missing_entries)}')    
                df_missing = df_vina[df_vina['vina_dir'].isnull()].copy()
                df_missing['structure'] = df_missing['chai_dir'].apply(generate_chai_structure_path)

            elif self.alternative_structure_for_vina == 'Boltz':
                log_boxed_note('Fallback to Boltz structures for docking due to missing AF2 structures. ' + f'Entries: {list(missing_entries)}')    
                df_missing = df_vina[df_vina['vina_dir'].isnull()].copy()
                df_missing['structure'] = df_missing['boltz_dir'].apply(generate_boltz_structure_path)

            df_missing_docked = df_missing << (Vina('Entry', 'structure', 'Sequence', 'substrate_smiles', 'substrate_name', 'catalytic_residues', vina_dir, self.num_threads))
            df_missing_docked.rename(columns = {'output_dir':'vina_dir_missing'}, inplace=True)

            # Merge both docking attempts
            df_vina_combined = pd.merge(
                df_vina, df_missing_docked[['Entry', 'vina_dir_missing']], on='Entry', how='left'
            )

            # Choose first attempt if available, otherwise fallback
            df_vina_combined['vina_dir'] = df_vina_combined.apply(
                lambda row: row['vina_dir'] if pd.notnull(row['vina_dir']) else row['vina_dir_missing'],
                axis=1
            )
            df_vina_combined.drop(columns=['vina_dir_missing'], inplace=True)
            df_vina = df_vina_combined.copy()
  
        df_vina.to_pickle(Path(self.output_dir)/'vina.pkl') 
        return df_vina

    @log_usage("Extract docking quality metrics")
    def _extract_docking_quality_metrics(self, df):
        log_subsection('Extracting docking quality metrics')
        # When Vina is enabled, filter on vina_dir; otherwise filter on boltz_dir
        if self.run_vina and 'vina_dir' in df.columns:
            df = df[df["vina_dir"].notna()].copy()
        else:
            df = df[df["boltz_dir"].notna()].copy()
        df_metrics = df << (DockingMetrics(input_dir = Path(self.output_dir), output_dir = Path(self.output_dir)) 
                        >> Save(Path(self.output_dir) / 'dockingmetrics.pkl'))
        return df_metrics


class Superimposition:
    def __init__(self, maxMatches, input_dir="pipeline_output", output_dir="pipeline_output",
                 include_vina: bool = False, num_threads=1,
                 run_fastrelax: bool = False,
                 fastrelax_mode: str = "ligand_focused",
                 fastrelax_top_k: int = 2,
                 fastrelax_drop_unrelaxed: bool = True,
                 fastrelax_shell_radius: float = 8.0,
                 fastrelax_constraint_weight: float = 1.0,
                 fastrelax_scorefunction: str = "ref2015",
                 ligand_resname: str = "LIG"):
        self.maxMatches = maxMatches
        self.include_vina = include_vina
        self.num_threads = num_threads
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.run_fastrelax = run_fastrelax
        self.fastrelax_mode = fastrelax_mode
        self.fastrelax_top_k = fastrelax_top_k
        self.fastrelax_drop_unrelaxed = fastrelax_drop_unrelaxed
        self.fastrelax_shell_radius = fastrelax_shell_radius
        self.fastrelax_constraint_weight = fastrelax_constraint_weight
        self.fastrelax_scorefunction = fastrelax_scorefunction
        self.ligand_resname = ligand_resname

    def run(self):
        
        log_section('Superimposition')
        log_subsection('Superimposing docked structures')
        df_prep = self._prepare_files_for_superimposition()
        if self.run_fastrelax:
            log_subsection('Relaxing top-ranked docked structures')
            df_prep = self._run_fastrelax(df_prep)
        df_sup = self._superimposition(df_prep)
        log_subsection('Calculating protein RMSDs')
        df_proteinRMSD_all, df_proteinRMSD  = self._proteinRMSD(df_sup)
        log_subsection('Calculating ligand RMSDs')
        df_ligandRMSD_all, df_ligandRMSD = self._ligandRMSD(df_proteinRMSD)
        return df_ligandRMSD


    def _run_fastrelax(self, df_prep):
        fastrelax_dir = Path(self.output_dir) / 'fastrelax'
        fastrelax_dir.mkdir(exist_ok=True, parents=True)
        step = FastRelax(
            output_dir=fastrelax_dir,
            mode=self.fastrelax_mode,
            top_k=self.fastrelax_top_k,
            drop_unrelaxed=self.fastrelax_drop_unrelaxed,
            shell_radius=self.fastrelax_shell_radius,
            constraint_weight=self.fastrelax_constraint_weight,
            scorefunction=self.fastrelax_scorefunction,
            num_threads=self.num_threads,
        )
        return step.execute(df_prep)

    def _prepare_files_for_superimposition(self):
        df_metrics = pd.read_pickle(Path(self.input_dir) / 'dockingmetrics.pkl')
        preparedfiles_dir = Path(self.output_dir) / 'preparedfiles_for_superimposition/'

        if self.include_vina:
            df_metrics << (PrepareVina('vina_dir', 'substrate_name', preparedfiles_dir)
                    >> PrepareChai('chai_dir', preparedfiles_dir, 1)
                    >> PrepareBoltz('boltz_dir', preparedfiles_dir, 1))
        else:
            df_metrics << (PrepareChai('chai_dir', preparedfiles_dir, 1)
                    >> PrepareBoltz('boltz_dir', preparedfiles_dir, 1))
        return df_metrics
    
    @log_usage("Superimposing structures")
    def _superimposition(self, df):                   
        output_sup_dir = Path(self.output_dir) / 'superimposed_structures'

        if self.include_vina:
            # Filter on rows where both vina and chai files are available
            df = df[df['vina_files_for_superimposition'].apply(valid_file_list)]
            df = df[df['chai_files_for_superimposition'].apply(valid_file_list)]

            # 3 pairwise superimpositions: vina-chai, vina-boltz, chai-boltz
            df_sup = df << (SuperimposeStructures('vina_files_for_superimposition', 'chai_files_for_superimposition', output_dir=output_sup_dir, name1='vina', name2='chai', num_threads=self.num_threads)
                    >> SuperimposeStructures('vina_files_for_superimposition', 'boltz_files_for_superimposition', output_dir=output_sup_dir, name1='vina', name2='boltz', num_threads=self.num_threads)
                    >> SuperimposeStructures('chai_files_for_superimposition', 'boltz_files_for_superimposition', output_dir=output_sup_dir, name1='chai', name2='boltz', num_threads=self.num_threads)
                    >> Save(Path(self.output_dir) / 'superimposedstructures.pkl'))
        else:
            # Only chai-boltz superimposition
            df = df[df['chai_files_for_superimposition'].apply(valid_file_list)]

            df_sup = df << (SuperimposeStructures('chai_files_for_superimposition', 'boltz_files_for_superimposition', output_dir=output_sup_dir, name1='chai', name2='boltz', num_threads=self.num_threads)
                    >> Save(Path(self.output_dir) / 'superimposedstructures.pkl'))
        return df_sup
    
    @log_usage("Calculating protein RMSD")
    def _proteinRMSD(self, df):  
        proteinRMSD_dir = Path(self.output_dir) / 'proteinRMSD'
        proteinRMSD_dir.mkdir(exist_ok=True, parents=True) 
        input_dir = Path(self.output_dir) / 'superimposed_structures'
        df_proteinRMSD_pairwise, df_proteinRMSD = df << (ProteinRMSD('Entry', input_dir = input_dir, output_dir = proteinRMSD_dir, visualize_heatmaps = True))
        df_proteinRMSD_pairwise.to_pickle(Path(self.output_dir) / 'proteinRMSD_pairwise.pkl')
        df_proteinRMSD.to_pickle(Path(self.output_dir)/ 'proteinRMSD.pkl')
        return df_proteinRMSD_pairwise, df_proteinRMSD

    @log_usage("Calculating ligand RMSD")
    def _ligandRMSD(self, df): 
        ligandRMSD_dir = Path(self.output_dir) / 'ligandRMSD'
        ligandRMSD_dir.mkdir(exist_ok=True, parents=True) 
        input_dir = Path(self.output_dir)  / 'superimposed_structures'

        df_ligandRMSD_pairwise, df_ligandRMSD = df << (LigandRMSD('Entry', input_dir = input_dir, output_dir = ligandRMSD_dir, visualize_heatmaps= True, maxMatches = self.maxMatches))
        df_ligandRMSD.to_pickle(Path(self.output_dir) / 'ligandRMSD_prior.pkl')
        df_ligandRMSD_w_metrics = extract_docking_metrics(df_ligandRMSD)
        df_ligandRMSD_w_metrics.to_pickle(Path(self.output_dir) / 'ligandRMSD.pkl')
        df_ligandRMSD_pairwise.to_pickle(Path(self.output_dir)/ 'ligandRMSD_pairwise.pkl')
        return df_ligandRMSD_pairwise, df_ligandRMSD_w_metrics


class GeometricFilters:
    def __init__(self, df, esterase=0, input_dir="superimposition", output_dir="geometricfiltering", num_threads=1):
        self.esterase = esterase
        self.df = df
        self.num_threads = num_threads
        if isinstance(df, str):  # Allow passing of a string
            self.df = pd.read_csv(df)
        else:
            self.df = df.copy()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


    def run(self):
        
        log_section('Running geometric filtering')
        log_subsection('Calculate catalytic residue/cofactor - ligand distances')
        df_filter = self._run_geometric_filtering()
        log_subsection('Calculate active site volume')
        df_ASvolume = self._active_site_volume(df_filter)
        log_subsection('Calculate ligand surface exposure')
        df_ASvolume = self._ligand_surface_exposure(df_ASvolume)
        log_subsection('Running PLIP to identify protein-ligand interactions')
        df_final = self._plip_interactions(df_ASvolume)

        out_final = Path(self.output_dir) / 'structural_features_final.pkl'
        df_final.to_pickle(out_final)

        log_boxed_note('Pipeline finished!')
        return df_final

    @log_usage("Geometric filtering")
    def _run_geometric_filtering(self):
        if self.esterase == 1: 
            df_geo_filter = self.df << (EsteraseGeometricFiltering(
                                            preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition',
                                            output_dir=self.output_dir)
                                    >> Save(Path(self.output_dir) / 'geometricfiltering.pkl'))
        else: 
            df_geo_filter = self.df << (GeneralGeometricFiltering(
                                            preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition',
                                            output_dir=self.output_dir)
                                    >> Save(Path(self.output_dir) / 'geometricfiltering.pkl'))
        return df_geo_filter

    @log_usage("Calculating active site volume")
    def _active_site_volume(self, df):
        fpocket_dir = Path(self.output_dir) / 'ASVolume'
        fpocket_dir.mkdir(exist_ok=True, parents=True)
        df_ASVolume = df << (Fpocket(preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition', output_dir = fpocket_dir)  
            >> Save(Path(self.output_dir) / 'ASvolume.pkl'))
        return df_ASVolume

    @log_usage("Calculating ligand surface exposure")
    def _ligand_surface_exposure(self, df):
        ligandSASA_dir = Path(self.output_dir) / 'LigandSASA'
        df_ligandSASA = df << (LigandSASA(input_dir = Path(self.input_dir)/ 'preparedfiles_for_superimposition', output_dir = ligandSASA_dir)
                            >> Save(Path(self.output_dir) / 'ligandSASA.pkl'))
        return df_ligandSASA
    
    @log_usage("Running PLIP for protein-ligand interactions")  
    def _plip_interactions(self, df):
        df_plip = df << (PLIP(input_dir = Path(self.input_dir) / 'preparedfiles_for_superimposition', output_dir = self.output_dir)
                        >> Save(Path(self.output_dir) / 'plip_interactions.pkl'))
        return df_plip


class Pipeline:
    """Backward-compatible adapter around the modular RunConfig + Runner engine.

    .. deprecated::
        Constructing ``Pipeline(df, boltz_cache_dir=..., ...)`` still works, but
        the legacy keyword arguments are now translated into a
        :class:`structurezyme.config.RunConfig` and executed by
        :class:`structurezyme.runner.Runner`.  New code should build a
        ``RunConfig`` and call ``Runner(cfg).run()`` directly.
    """

    def __init__(self,
                df, 
                boltz_cache_dir: str,
                max_matches: int = 1000,
                esterase: int = 0,
                metagenomic_enzymes: int = 0,
                skip_catalytic_residue_prediction: bool = False,
                run_vina: bool = False,
                alternative_structure_for_vina: str = 'Boltz', 
                use_msa_server: bool = True,
                num_threads: int = 1,
                squidly_dir: Union[str, Path] = '',
                base_output_dir: Union[str, Path] = "pipeline_output",
                squidly_model_size: str = '3B',
                squidly_as_threshold: float | None = None,
                squidly_num_threads: int | None = None,
                run_placer: bool = False,
                placer_predict_ligand: str | None = None,
                placer_nsamples: int = 50,
                placer_rerank: str = "prmsd",
                placer_env_path: str = "/mnt/labs/data/mora/software/PLACER/env",
                run_fastrelax: bool = False,
                fastrelax_mode: str = "ligand_focused",
                fastrelax_top_k: int = 2,
                fastrelax_drop_unrelaxed: bool = True,
                fastrelax_shell_radius: float = 8.0,
                fastrelax_constraint_weight: float = 1.0,
                fastrelax_scorefunction: str = "ref2015",
                ligand_resname: str = "LIG",
                ):

        import warnings
        from datetime import datetime
        from structurezyme.config import (
            RunConfig, PathsConfig, RuntimeConfig, StepsConfig, StepConfig,
        )
        from structurezyme.paths import run_dir, RunLayout

        warnings.warn(
            "structurezyme.pipeline.Pipeline is deprecated; build a "
            "structurezyme.config.RunConfig and run it with "
            "structurezyme.runner.Runner instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if run_placer and placer_predict_ligand is None:
            raise ValueError(
                "run_placer=True requires placer_predict_ligand (e.g. 'A-HEM-154')"
            )

        self.df = df.copy()

        # Options shared by several steps.
        squidly_opts = dict(
            skip_catalytic_residue_prediction=skip_catalytic_residue_prediction,
            squidly_model_size=squidly_model_size,
            squidly_as_threshold=squidly_as_threshold,
            squidly_num_threads=squidly_num_threads,
            squidly_dir=str(squidly_dir),
        )
        vina_opts = dict(
            metagenomic_enzymes=metagenomic_enzymes,
            alternative_structure_for_vina=alternative_structure_for_vina,
        )
        fastrelax_opts = dict(
            fastrelax_mode=fastrelax_mode,
            fastrelax_top_k=fastrelax_top_k,
            fastrelax_drop_unrelaxed=fastrelax_drop_unrelaxed,
            fastrelax_shell_radius=fastrelax_shell_radius,
            fastrelax_constraint_weight=fastrelax_constraint_weight,
            fastrelax_scorefunction=fastrelax_scorefunction,
        )
        placer_opts = dict(
            placer_predict_ligand=placer_predict_ligand,
            placer_nsamples=placer_nsamples,
            placer_rerank=placer_rerank,
            placer_env_path=placer_env_path,
        )

        steps = StepsConfig(
            squidly=StepConfig(enabled=True, **squidly_opts),
            boltz=StepConfig(enabled=True, use_msa_server=use_msa_server),
            vina=StepConfig(enabled=run_vina, **vina_opts),
            fastrelax=StepConfig(enabled=run_fastrelax, **fastrelax_opts),
            superimpose=StepConfig(enabled=True),
            ligand_rmsd=StepConfig(enabled=True, max_matches=max_matches),
            geometric_filter=StepConfig(enabled=True, esterase=esterase),
            placer=StepConfig(enabled=run_placer, **placer_opts),
        )

        # Explicit RuntimeConfig so each Pipeline instance gets a *fresh* run_id
        # (the RunConfig class-level default freezes run_id at import time).
        runtime = RuntimeConfig(
            run_id=datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
            num_threads=num_threads,
        )
        self.config = RunConfig(
            paths=PathsConfig(
                output_root=str(base_output_dir),
                boltz_cache_dir=str(boltz_cache_dir),
                squidly_weights_dir=str(squidly_dir) or None,
                placer_env_path=placer_env_path,
            ),
            runtime=runtime,
            steps=steps,
        )

        # Build the run layout and seed the input DataFrame so the first step
        # (squidly) can read it from checkpoints/_input.pkl.
        self.layout = RunLayout(run_dir(self.config.paths.output_root,
                                        self.config.runtime.user,
                                        self.config.runtime.run_id))
        self.layout.create()
        self.df.to_pickle(self.layout.checkpoint_path("_input"))

    def run(self):
        """Execute the run via the modular Runner (config built in __init__)."""
        from structurezyme.runner import Runner

        Runner(self.config).run()
