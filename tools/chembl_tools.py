from langchain_core.tools import tool

from query_chembl import (
    get_admet_properties,
    get_structural_physicochemical_properties,
)


@tool
def get_admet_properties_tool(drug_name: str) -> dict:
    """Return ADMET-related drug properties from ChEMBL."""
    return get_admet_properties(drug_name) or {
        "error": f"No ChEMBL entry found for {drug_name!r}."
    }


@tool
def get_structural_physicochemical_properties_tool(drug_name: str) -> dict:
    """Return structural and physicochemical drug properties from ChEMBL."""
    return get_structural_physicochemical_properties(drug_name) or {
        "error": f"No ChEMBL entry found for {drug_name!r}."
    }
