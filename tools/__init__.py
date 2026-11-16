from tools.chembl_tools import (
    get_admet_properties_tool,
    get_structural_physicochemical_properties_tool,
)
from tools.clinicaltrials_tools import get_clinical_trial_tool, clinicaltrials_gov_search
from tools.europepmc_tools import get_publication_by_pmid_tool, europe_pmc_search, get_publication_full_text_tool
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

DISEASE_TARGET_TOOLS.extend([get_clinical_trial_tool, get_publication_by_pmid_tool,
                             clinicaltrials_gov_search, europe_pmc_search, get_publication_full_text_tool])

DRUG_PROPERTY_TOOLS = [
    get_admet_properties_tool,
    get_structural_physicochemical_properties_tool,
]
