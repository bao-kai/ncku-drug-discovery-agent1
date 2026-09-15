from langchain_core.tools import tool

from query_opentargets import (
    get_disease_associated_targets,
    get_disease_target_evidence,
    get_efo_id,
    get_ensembl_id,
    get_known_drugs_associated_with_target,
)


@tool
def get_efo_id_tool(disease_name: str) -> str:
    """Look up the EFO or MONDO ontology ID for a disease name."""
    return get_efo_id(disease_name) or f"No disease ID found for {disease_name!r}."


@tool
def get_ensembl_id_tool(gene_symbol: str) -> str:
    """Look up the Ensembl gene ID for a target symbol or name."""
    return get_ensembl_id(gene_symbol) or f"No target ID found for {gene_symbol!r}."


@tool
def get_disease_associated_targets_tool(efo_id: str) -> list[dict]:
    """Return the top ten targets associated with a disease ontology ID."""
    return get_disease_associated_targets(efo_id)


@tool
def get_disease_target_evidence_tool(
    disease_name: str, target_name: str
) -> dict:
    """Return traceable Open Targets scores for one disease-target pair."""
    return get_disease_target_evidence(disease_name, target_name)


@tool
def get_known_drugs_associated_with_target_tool(
    ensembl_id: str, efo_id: str
) -> list[dict]:
    """Return known drugs for a target, preferring the specified disease."""
    return get_known_drugs_associated_with_target(ensembl_id, efo_id)
