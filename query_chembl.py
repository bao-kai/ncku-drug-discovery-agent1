"""ChEMBL REST queries retained from the original project with timeouts."""

from __future__ import annotations

import requests


CHEMBL_ENDPOINT = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT_SECONDS = 30


def _get(path: str, params: dict | None = None) -> dict:
    response = requests.get(
        f"{CHEMBL_ENDPOINT}/{path}", params=params, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def _find_chembl_id(drug_name: str) -> tuple[str, str] | tuple[None, None]:
    for params in [
        {"pref_name__iexact": drug_name.upper(), "format": "json", "limit": 1},
        {
            "molecule_synonyms__molecule_synonym__iexact": drug_name,
            "format": "json",
            "limit": 1,
        },
    ]:
        molecules = _get("molecule", params).get("molecules", [])
        if molecules:
            molecule = molecules[0]
            return (
                molecule["molecule_chembl_id"],
                molecule.get("pref_name") or drug_name,
            )
    return None, None


def get_admet_properties(drug_name: str) -> dict | None:
    chembl_id, preferred_name = _find_chembl_id(drug_name)
    if not chembl_id:
        return None
    molecule = _get(f"molecule/{chembl_id}", {"format": "json"})
    props = molecule.get("molecule_properties") or {}
    return {
        "chembl_id": chembl_id,
        "drug_name": preferred_name,
        "max_phase": molecule.get("max_phase"),
        "oral": molecule.get("oral"),
        "parenteral": molecule.get("parenteral"),
        "topical": molecule.get("topical"),
        "black_box_warning": molecule.get("black_box_warning"),
        "withdrawn": molecule.get("withdrawn_flag"),
        "alogp": props.get("alogp"),
        "psa": props.get("psa"),
        "hbd": props.get("hbd"),
        "hba": props.get("hba"),
        "num_ro5_violations": props.get("num_ro5_violations"),
        "ro3_pass": props.get("ro3_pass"),
        "qed_weighted": props.get("qed_weighted"),
        "np_likeness_score": props.get("np_likeness_score"),
        "rtb": props.get("rtb"),
    }


def get_structural_physicochemical_properties(drug_name: str) -> dict | None:
    chembl_id, preferred_name = _find_chembl_id(drug_name)
    if not chembl_id:
        return None
    molecule = _get(f"molecule/{chembl_id}", {"format": "json"})
    props = molecule.get("molecule_properties") or {}
    structures = molecule.get("molecule_structures") or {}
    return {
        "chembl_id": chembl_id,
        "drug_name": preferred_name,
        "molecule_type": molecule.get("molecule_type"),
        "molecular_formula": props.get("full_molformula"),
        "molecular_weight": props.get("full_mwt"),
        "mw_freebase": props.get("mw_freebase"),
        "heavy_atoms": props.get("heavy_atoms"),
        "aromatic_rings": props.get("aromatic_rings"),
        "rotatable_bonds": props.get("rtb"),
        "chirality": molecule.get("chirality"),
        "canonical_smiles": structures.get("canonical_smiles"),
        "standard_inchi_key": structures.get("standard_inchi_key"),
        "first_approval": molecule.get("first_approval"),
        "atc_classifications": molecule.get("atc_classifications"),
    }
