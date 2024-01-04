from typing import Optional, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
from aiida.orm import TrajectoryData
from ase import units
from tqdm.auto import tqdm
import joblib

from aiida_utils.load_data import load_data
from pymatgen.core import Structure


def load_and_preprocess_aiida_data(archive_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from AiiDA archive and return a pandas DataFrame with only basic types,
    pymatgen structures & numpy arrays (i.e., no AiiDA nodes).
    """

    args = [archive_file] if archive_file else []
    data = load_data(*args)
    columns_of_interest = {
        "stru": "starting_structure",
        "traj": "trajectory",
        "temp": "temperature",
        "diff_mean": "diffusion_mean_cm2_s",
        "diff_std": "diffusion_std_cm2_s",
        "diff_sem": "diffusion_sem_cm2_s",
        "stru_label": "label",
        "stru_db": "src_database",
        "stru_id": "src_id"
    }
    data_prep = pd.DataFrame(
        {
            target_column: data[src_column]
            for src_column, target_column in columns_of_interest.items()
        }
    )
    data_prep["starting_structure"] = data_prep["starting_structure"].apply(lambda x: x.get_pymatgen())
    data_prep["first_frame_structure"] = data_prep["trajectory"].apply(lambda x: x.get_step_structure(0).get_pymatgen())
    data_prep["trajectory"] = data_prep["trajectory"].apply(lambda x: x.get_positions())

    return data_prep


def make_csv(
    output_file: str, *,
    archive_file: Optional[str] = None,
    include_trajectories: bool = False,
) -> None:
    data = load_and_preprocess_aiida_data(archive_file=archive_file)
    data["starting_structure"] = data["starting_structure"].apply(lambda x: x.to_json())
    data["first_frame_structure"] = data["first_frame_structure"].apply(lambda x: x.to_json())

    if include_trajectories:
        data["trajectory"] = data["trajectory"].apply(lambda x: json.dumps(x.tolist()))
    else:
        data.drop("trajectory", axis=1, inplace=True)

    data.to_csv(output_file, index=False)

def read_csv(input_file: str) -> pd.DataFrame:
    data = pd.read_csv(input_file)
    data["starting_structure"] = data["starting_structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
    data["first_frame_structure"] = data["first_frame_structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
    if "trajectory" in data.columns:
        data["trajectory"] = data["trajectory"].apply(lambda x: np.array(json.loads(x)))

    return data

def make_joblib_pickle(
    export_path: str,
    archive_file: Optional[str] = None,
):
    def process_trajectory_node(traj: TrajectoryData) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        structure_json = traj.get_step_structure(0).get_pymatgen().to_json()
        positions = traj.get_array("positions")
        energies = traj.get_array("potential_energies") * units.Ry
        forces = traj.get_array("forces") * units.Ry / units.Bohr
        return structure_json, positions, energies, forces

    export_path = Path(export_path)
    assert not export_path.exists()

    load_data_args = {}
    if archive_file is not None:
        load_data_args["archive_file"] = archive_file
    data = load_data(**load_data_args)
    export_path.mkdir()

    for i, row in tqdm(data.iterrows(), total=len(data), desc="Writing trajectories to disc"):
        traj_data = (
            (row["stru_db"], row["stru_id"], row["temp"]),
            process_trajectory_node(row["traj"])
        )
        joblib.dump(traj_data, export_path / f"traj_data_{i:03d}.joblib.pkl")
