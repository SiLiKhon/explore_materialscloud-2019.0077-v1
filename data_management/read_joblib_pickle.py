from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import joblib
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from tqdm.auto import tqdm


def load_file(filename: Path) -> Tuple[List[Atoms], np.ndarray, np.ndarray, str, str]:
    (
        (stru_db, stru_id),
        (structure_json, positions, energies, forces),
    ) = joblib.load(filename, mmap_mode="c")
    stru = AseAtomsAdaptor.get_atoms(Structure.from_str(structure_json, "json"))
    assert len(positions) == len(energies) == len(forces)
    structures = []
    for pos in positions:
        stru_i = stru.copy()
        stru_i.positions = pos
        structures.append(stru_i)

    return (
        structures, energies, forces, stru_db, stru_id
    )

def read_joblib_pickle(import_path: str) -> Dict[str, Any]:
    import_path = Path(import_path)
    assert import_path.exists()

    files = sorted(import_path.glob("*.joblib.pkl"))
    assert len(files) == 200

    result = {}
    for f in tqdm(files):
        structures, energies, forces, stru_db, stru_id = load_file(f)
        key = (stru_db, stru_id, str(structures[0].symbols))
        if key not in result:
            result[key] = (structures, energies, forces)
        else:
            result[key] = (
                result[key][0] + structures,
                np.concatenate([result[key][1], energies], axis=0),
                np.concatenate([result[key][2], forces], axis=0),
            )

    return result
