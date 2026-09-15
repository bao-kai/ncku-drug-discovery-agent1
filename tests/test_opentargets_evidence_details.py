from unittest.mock import patch

from query_opentargets import (
    DISEASE_TARGET_PAIR_QUERY,
    _named_type,
    get_disease_target_evidence,
    get_datasource_evidence_records,
)


def test_named_type_unwraps_non_null_list():
    ref = {
        "kind": "NON_NULL",
        "name": None,
        "ofType": {
            "kind": "LIST",
            "name": None,
            "ofType": {"kind": "SCALAR", "name": "String", "ofType": None},
        },
    }
    assert _named_type(ref) == ("SCALAR", "String")


@patch(
    "query_opentargets._selection_for_type",
    side_effect=lambda _name, depth, **_kwargs: (
        ["id", "score", "drugId", "clinicalPhase", "urls { url }"]
        if depth == 2
        else ["id", "score", "drugId", "clinicalPhase"]
    ),
)
@patch("query_opentargets._graphql")
def test_clinical_precedence_records_include_count_and_size_estimate(graphql, _sel):
    graphql.return_value = {
        "disease": {
            "evidences": {
                "count": 3,
                "rows": [
                    {
                        "id": "e1",
                        "score": 0.7,
                        "drugId": "CHEMBL1",
                        "clinicalPhase": 3,
                    }
                ],
            }
        }
    }
    result = get_datasource_evidence_records(
        "MONDO_1", "ENSG1", "clinical_precedence", max_records=1
    )
    assert result["total_evidence_count"] == 3
    assert result["fetched_evidence_count"] == 1
    assert result["truncated"] is True
    assert result["estimated_complete_size_bytes"] > 0
    assert result["records"][0]["drugId"] == "CHEMBL1"


@patch("query_opentargets._graphql")
@patch("query_opentargets._search")
def test_pai1_pair_query_uses_verified_ids_instead_of_top_100(search, graphql):
    search.side_effect = [
        {"id": "MONDO_0005192", "name": "exocrine pancreatic carcinoma"},
        {"id": "ENSG00000106366", "name": "PAI-1"},
    ]
    graphql.return_value = {
        "disease": {
            "id": "MONDO_0005192",
            "name": "exocrine pancreatic carcinoma",
            "associatedTargets": {
                "count": 1,
                "rows": [
                    {
                        "target": {
                            "id": "ENSG00000106366",
                            "approvedSymbol": "SERPINE1",
                            "approvedName": "serpin family E member 1",
                        },
                        "score": 0.09136039591803091,
                        "datasourceScores": [
                            {"id": "europepmc", "score": 0.7399592256178349}
                        ],
                    }
                ],
            },
        }
    }

    result = get_disease_target_evidence("Pancreatic cancer", "PAI-1")

    graphql.assert_called_once_with(
        DISEASE_TARGET_PAIR_QUERY,
        {"efoId": "MONDO_0005192", "ensemblId": "ENSG00000106366"},
    )
    assert result["target_symbol"] == "SERPINE1"
    assert result["overall_association_score"] == 0.09136039591803091
    assert result["datasource_scores"][0]["datasource_id"] == "europepmc"
    assert result["fetched_target_count"] == 1
    assert result["rank_within_fetched_targets"] is None
    assert "exact pair" in result["limitations"][0]
