from argparse import ArgumentParser
from pathlib import Path

from pymatgen.core import Structure
import joblib
import numpy as np
from tqdm.auto import tqdm
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.singlepoint import SinglePointCalculator


def _get_time_step(times: np.ndarray) -> float:
    dt = np.diff(times)
    dt = dt[dt > 0]
    dt = np.unique(np.round(dt, 4))
    (dt,) = dt
    return float(dt)


def process_single_file(
    input_joblib_pickle_file_name: str,
    output_ase_traj_file_name: str,
) -> None:
    (
        (stru_db, stru_id, temperature),
        (stru, xyz, ee, ff, times_ps),
    ) = joblib.load(input_joblib_pickle_file_name)
    stru = Structure.from_str(stru, "json")
    dt = _get_time_step(times_ps)

    assert len(xyz) == len(ee) == len(ff) == len(times_ps)
    assert xyz.shape == ff.shape
    atoms = AseAtomsAdaptor.get_atoms(stru)
    atoms = [atoms.copy() for _ in range(len(ee))]

    for ats_i, positions, energy, forces in zip(atoms, xyz, ee, ff):
        ats_i.positions = positions
        props = {
            "energy": energy,
            "forces": forces,
        }
        calc = SinglePointCalculator(
            ats_i, **props,
        )
        calc.implemented_properties = list(props.keys())
        ats_i.calc = calc

    atoms[0].info["time_step_ps"] = dt
    atoms[0].info["database"] = stru_db
    atoms[0].info["id"] = stru_id
    atoms[0].info["temperature"] = temperature

    write(output_ase_traj_file_name, atoms)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--joblib-in", "-i", type=str, required=True)
    parser.add_argument("--ase-out", "-o", type=str, required=True)

    args = parser.parse_args()
    in_path = Path(args.joblib_in)
    out_path = Path(args.ase_out)

    if not in_path.is_dir():
        raise ValueError("input path should be a valid existing path")

    if out_path.exists():
        raise ValueError("out path should not exist")

    in_files = sorted(in_path.glob("*.joblib.pkl"))

    if len(in_files) == 0:
        raise ValueError("provided path contains no '*.joblib.pkl' files")

    out_path.mkdir()

    for in_file in tqdm(in_files):
        out_file = out_path / f"{in_file.stem}.traj"
        if out_file.exists():
            raise ValueError(f"{out_file} exists already (wut?)")

        process_single_file(
            input_joblib_pickle_file_name=in_file.as_posix(),
            output_ase_traj_file_name=out_file.as_posix(),
        )
