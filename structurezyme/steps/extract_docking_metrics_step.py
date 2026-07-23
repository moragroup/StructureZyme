import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import json

from structurezyme.steps.step import Step

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




def parse_vina_output(file_path):
    docking_results = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    parsing = False
    for line in lines:
        line = line.strip()

        if line.startswith("mode |"):  # start of the table
            parsing = True
            continue

        if parsing:
            if not line or not line[0].isdigit():
                continue  # skip headers or footers

            parts = line.split()
            try:
                mode = int(parts[0])
                affinity = float(parts[1])
                docking_results[mode] = affinity
            except Exception as e:
                logger.warning(f"Could not parse line: '{line}' — {e}")
                continue
    return docking_results


def _vina_log_path(vina_dir, entry, substrate_name) -> Path:
    """Build the path to the Vina log file written by `dock_vina()`.

    Historical note: the column name `vina_dir` is misleading. `Vina._execute`
    (`structurezyme/steps/dock_vina_step.py`) appends `str(pdb_path)` — the
    receptor PDB *file* path (`<label_dir>/<Entry>.pdb`) — to
    `output_filenames`, which is then renamed to `vina_dir` in
    `run_vina`/`_run_vina`. `PrepareVina` handles this correctly via
    `vina_path.parent` / `vina_path.stem`. This helper mirrors that convention:
    take the parent directory of whatever `vina_dir` points at so the log
    lookup works whether the caller passes a file or a directory.
    """
    p = Path(vina_dir)
    label_dir = p.parent if p.is_file() or p.suffix else p
    return label_dir / f"{entry}-{substrate_name}_log.txt"


def extract_chai_metrics(npz_path):
    '''
    Extract chai metrics from npz files.
    '''
    data = np.load(npz_path)
    
    return {
        "filename": npz_path.name,
        "aggregate_score": float(data["aggregate_score"][0]),
        "ptm": float(data["ptm"][0]),
        "iptm": float(data["iptm"][0]),
        "per_chain_ptm": data["per_chain_ptm"][0].tolist(),
        "per_chain_pair_iptm": data["per_chain_pair_iptm"][0].tolist(),
        "has_inter_chain_clashes": bool(data["has_inter_chain_clashes"][0]),
        "chain_chain_clashes": data["chain_chain_clashes"][0].tolist()
    }


def round_sig(x, sig=5):
    """
    Round to significant digits.
    """
    if isinstance(x, float):
        return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0.0
    if isinstance(x, list):
        return [round_sig(v, sig) for v in x]
    if isinstance(x, (np.ndarray, tuple)):
        return [round_sig(float(v), sig) for v in x]
    return x


def extract_boltz2_metrics_from_json(conf_json_path):
    with open(conf_json_path, "r") as f:
        data = json.load(f)

    return {
        "confidence_score": round(data.get("confidence_score", 0.0), 5),
        "ptm": round(data.get("ptm", 0.0), 5),
        "iptm": round(data.get("iptm", 0.0), 5),
        "ligand_iptm": round(data.get("ligand_iptm", 0.0), 5),
        "protein_iptm": round(data.get("protein_iptm", 0.0), 5),
        "complex_plddt": round(data.get("complex_plddt", 0.0), 5),
        "complex_iplddt": round(data.get("complex_iplddt", 0.0), 5),
        "complex_pde": round(data.get("complex_pde", 0.0), 5),
        "complex_ipde": round(data.get("complex_ipde", 0.0), 5),
        "chains_ptm": data.get("chains_ptm", {}),
        "pair_chains_iptm": data.get("pair_chains_iptm", {}),
    }


def extract_boltz2_affinity(conf_json_path):
    with open(conf_json_path, "r") as f:
        data = json.load(f)

    return {
        "affinity_pred_value": round(data.get("affinity_pred_value", 0.0), 5),
        "affinity_probability_binary": round(data.get("affinity_probability_binary", 0.0), 5),
        "affinity_pred_value1": round(data.get("affinity_pred_value1", 0.0), 5),
        "affinity_probability_binary1": round(data.get("affinity_probability_binary1", 0.0), 5),
        "affinity_pred_value2": round(data.get("affinity_pred_value2", 0.0), 5),
        "affinity_probability_binary2": round(data.get("affinity_probability_binary2", 0.0), 5),
    }


class DockingMetrics(Step):
    def __init__(self, input_dir='pipeline_output/docking', output_dir='pipeline_output/docking'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.input_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            ligand_name = row['substrate_name']

            row_result = {}

            # ---Exract chai docking metrics---
            chai_dir = Path(row['chai_dir']) / 'chai'
            npz_files = list(chai_dir.rglob("*.npz"))

            # Initialize per-metric dicts
            aggregate_dict = {}
            ptm_dict = {}
            iptm_dict = {}
            per_chain_ptm_dict = {}
            per_chain_pair_iptm_dict = {}
            has_clashes_dict = {}
            chain_chain_clashes_dict = {}

            for npz_file in npz_files:
                try:
                    metrics = extract_chai_metrics(npz_file)
                    fname = npz_file.stem  # no .npz extension

                    aggregate_dict[fname] = round_sig(metrics["aggregate_score"])
                    ptm_dict[fname] = round_sig(metrics["ptm"])
                    iptm_dict[fname] = round_sig(metrics["iptm"])
                    per_chain_ptm_dict[fname] = round_sig(metrics["per_chain_ptm"])
                    per_chain_pair_iptm_dict[fname] = round_sig(metrics["per_chain_pair_iptm"])
                    has_clashes_dict[fname] = metrics["has_inter_chain_clashes"]  
                    chain_chain_clashes_dict[fname] = round_sig(metrics["chain_chain_clashes"])

                except Exception as e:
                    logger.warning(f"Skipped {npz_file.name}: {e}")
                    continue

            # Add all dicts to row
            row_result.update({
                'chai_aggregate_score': aggregate_dict,
                'chai_ptm': ptm_dict,
                'chai_iptm': iptm_dict,
                'chai_per_chain_ptm': per_chain_ptm_dict,
                'chai_per_chain_pair_iptm': per_chain_pair_iptm_dict,
                'chai_has_clashes': has_clashes_dict,
                'chai_chain_chain_clashes': chain_chain_clashes_dict,
            })

            # ---Exract boltz2 docking metrics---
            boltz2_dir = Path(row['boltz_dir']) / f'boltz_results_{entry_name}' / 'predictions' / entry_name

            # Initialize per-metric dicts                        
            boltz2_metrics_per_model = {
                "boltz2_confidence_score": {},
                "boltz2_ptm": {},
                "boltz2_iptm": {},
                "boltz2_ligand_iptm": {},
                "boltz2_protein_iptm": {},
                "boltz2_complex_plddt": {},
                "boltz2_complex_iplddt": {},
                "boltz2_complex_pde": {},
                "boltz2_complex_ipde": {},
                "boltz2_chains_ptm": {},
                "boltz2_pair_chains_iptm": {}, 
                "boltz2_affinity_pred_value": {}, 
                "boltz2_affinity_probability_binary":{}, 
                "boltz2_affinity_pred_value1": {}, 
                "boltz2_affinity_probability_binary1": {}, 
                "boltz2_affinity_pred_value2": {}, 
                "boltz2_affinity_probability_binary2": {}
            }

            # extract folding and docking metrics
            for json_file in sorted(boltz2_dir.glob("confidence_*.json")):
                model_name = json_file.stem.replace("confidence_", "").replace(".json", "")
                try:
                    metrics = extract_boltz2_metrics_from_json(json_file)

                    boltz2_metrics_per_model["boltz2_confidence_score"][model_name] = metrics["confidence_score"]
                    boltz2_metrics_per_model["boltz2_ptm"][model_name] = metrics["ptm"]
                    boltz2_metrics_per_model["boltz2_iptm"][model_name] = metrics["iptm"]
                    boltz2_metrics_per_model["boltz2_ligand_iptm"][model_name] = metrics["ligand_iptm"]
                    boltz2_metrics_per_model["boltz2_protein_iptm"][model_name] = metrics["protein_iptm"]
                    boltz2_metrics_per_model["boltz2_complex_plddt"][model_name] = metrics["complex_plddt"]
                    boltz2_metrics_per_model["boltz2_complex_iplddt"][model_name] = metrics["complex_iplddt"]
                    boltz2_metrics_per_model["boltz2_complex_pde"][model_name] = metrics["complex_pde"]
                    boltz2_metrics_per_model["boltz2_complex_ipde"][model_name] = metrics["complex_ipde"]
                    boltz2_metrics_per_model["boltz2_chains_ptm"][model_name] = metrics["chains_ptm"]
                    boltz2_metrics_per_model["boltz2_pair_chains_iptm"][model_name] = metrics["pair_chains_iptm"]

                except Exception as e:
                    logger.warning(f"failed to parse {json_file.name}: {e}")

            # extract affinity
            for json_file in sorted(boltz2_dir.glob("affinity_*.json")):
                model_name = json_file.stem.replace("confidence_", "").replace(".json", "")
                try: 
                    metrics = extract_boltz2_affinity(json_file)

                    boltz2_metrics_per_model["boltz2_affinity_pred_value"][model_name] = metrics["affinity_pred_value"]
                    boltz2_metrics_per_model["boltz2_affinity_probability_binary"][model_name] = metrics["affinity_probability_binary"]
                    boltz2_metrics_per_model["boltz2_affinity_pred_value1"][model_name] = metrics["affinity_pred_value1"]
                    boltz2_metrics_per_model["boltz2_affinity_probability_binary1"][model_name] = metrics["affinity_probability_binary1"]
                    boltz2_metrics_per_model["boltz2_affinity_pred_value2"][model_name] = metrics["affinity_pred_value2"]
                    boltz2_metrics_per_model["boltz2_affinity_probability_binary2"][model_name] = metrics["affinity_probability_binary2"]

                except Exception as e:
                    logger.warning(f"failed to parse {json_file.name}: {e}")

            row_result.update(boltz2_metrics_per_model)

            # ---Extract vina docking metrics---
            vina_affinities = {}
            vina_dir_val = row.get('vina_dir')
            # pd.notna FIRST: with `x and pd.notna(x)` python evaluates `x`
            # first, and if x is pd.NA the short-circuit path raises
            # `TypeError: boolean value of NA is ambiguous` before pd.notna
            # gets a chance. Runs where vina is skipped for every row (e.g.
            # halogenases where squidly returns no residues) populate
            # `vina_dir` with pd.NA, which triggered the crash. pd.notna
            # returns a plain bool so ordering it first makes the guard safe.
            if pd.notna(vina_dir_val) and vina_dir_val:
                log_path = _vina_log_path(vina_dir_val, entry_name, ligand_name)
                if log_path.exists():
                    try:
                        vina_affinities = parse_vina_output(log_path)
                    except Exception as e:
                        logger.warning(f"Failed to parse vina log {log_path}: {e}")
            row_result['vina_affinities'] = vina_affinities

            results.append(row_result)

        return results

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            logger.warning("No output directory provided")
            return df

        results = self.__execute(df, self.output_dir)
        results_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        return output_df

