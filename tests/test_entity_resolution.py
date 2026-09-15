from unittest.mock import patch

from query_opentargets import EntityResolutionError, resolve_entity


@patch("query_opentargets.search_entity_candidates")
def test_ambiguous_disease_requires_explicit_id(search):
    search.return_value = [
        {"id": "MONDO_0005192", "name": "exocrine pancreatic carcinoma"},
        {"id": "EFO_OTHER", "name": "pancreatic neoplasm"},
    ]
    try:
        resolve_entity("pancreatic cancer", "disease")
    except EntityResolutionError as exc:
        assert len(exc.candidates) == 2
    else:
        raise AssertionError("Ambiguous disease was selected automatically")


@patch("query_opentargets.search_entity_candidates")
def test_exact_target_name_can_resolve(search):
    search.return_value = [
        {"id": "ENSG00000133703", "name": "KRAS"},
        {"id": "OTHER", "name": "KRAS-related protein"},
    ]
    result = resolve_entity("KRAS", "target")
    assert result["id"] == "ENSG00000133703"
    assert result["symbol"] == "KRAS"
