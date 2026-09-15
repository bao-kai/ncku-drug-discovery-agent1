from tools.chembl_tools import (
    get_admet_properties_tool,
    get_structural_physicochemical_properties_tool,
)
from tools.opentargets_tools import (
    get_disease_associated_targets_tool,
    get_disease_target_evidence_tool,
    get_efo_id_tool,
    get_ensembl_id_tool,
    get_known_drugs_associated_with_target_tool,
)

DISEASE_TARGET_TOOLS = [
    get_efo_id_tool,
    get_ensembl_id_tool,
    get_disease_associated_targets_tool,
    get_disease_target_evidence_tool,
    get_known_drugs_associated_with_target_tool,
]

DRUG_PROPERTY_TOOLS = [
    get_admet_properties_tool,
    get_structural_physicochemical_properties_tool,
]
