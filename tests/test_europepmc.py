from unittest.mock import Mock
import pytest
import requests
import query_europepmc as epmc
from tools import DISEASE_TARGET_TOOLS


def test_exact_pmid_preserves_native_abstract_and_provenance(monkeypatch):
    row = {'id':'33865446','source':'MED','abstractText':'Original abstract','title':'Original title'}
    get = Mock(return_value=Mock(json=lambda: {'hitCount':1,'resultList':{'result':[row]}}))
    monkeypatch.setattr(epmc.requests,'get',get)
    result = epmc.get_publication_by_pmid('33865446')
    assert result.status == 'success' and result.raw_record == row
    assert 'EXT_ID' in result.source_url and result.retrieved_at
    assert get.call_args.kwargs['params']['query'] == 'EXT_ID:33865446 AND SRC:MED'


@pytest.mark.parametrize('payload,status', [
    ({'hitCount':0,'resultList':{'result':[]}},'not_found'),
    ({'hitCount':1,'resultList':{'result':[{'id':'2','source':'MED'}]}},'error'),
    ({'hitCount':1,'resultList':{'result':[{'id':'1','source':'PMC'}]}},'error'),
    ({'hitCount':1,'resultList':{'result':['invalid']}},'error'),
    ({},'error'),
])
def test_empty_malformed_and_wrong_identity_are_distinct(monkeypatch,payload,status):
    monkeypatch.setattr(epmc.requests,'get',Mock(return_value=Mock(json=lambda:payload)))
    assert epmc.get_publication_by_pmid('1').status == status


def test_transport_error_is_not_missing_evidence(monkeypatch):
    monkeypatch.setattr(epmc.requests,'get',Mock(side_effect=requests.Timeout('timeout')))
    assert epmc.get_publication_by_pmid('1').status == 'error'
    with pytest.raises(ValueError):
        epmc.get_publication_by_pmid('1 OR cancer')


def test_both_exact_lookup_tools_are_callable_from_registered_worker_tools():
    names={t.name for t in DISEASE_TARGET_TOOLS}
    assert {'get_clinical_trial_tool','get_publication_by_pmid_tool'} <= names
