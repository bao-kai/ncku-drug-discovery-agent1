"""Exact-ID trial lookup tool; not automatically added to a Planner's tools."""

from langchain_core.tools import tool

from query_clinicaltrials import get_clinical_trial, search_clinical_trials


@tool
def clinicaltrials_gov_search(condition: str = '', intervention: str = '', term: str = '',
                             overall_statuses: list[str] | None = None, page_token: str | None = None) -> dict:
    """Search up to 20 native studies per call. Includes recorded phases, interventions, status/stop reasons and posted results if present. Return truncation and continuation; target keywords alone do not confirm drug-target relationships. Continue only when the Plan/follow-up calls for another page."""
    return search_clinical_trials(condition, intervention, term, overall_statuses, page_token)


@tool
def get_clinical_trial_tool(nct_id: str) -> dict:
    """Fetch one ClinicalTrials.gov study by NCT ID with raw data and provenance."""
    return get_clinical_trial(nct_id).model_dump(mode="json")
