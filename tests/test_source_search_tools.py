from unittest.mock import Mock
import pytest
import query_europepmc as epmc
import query_clinicaltrials as ct
from tools import DISEASE_TARGET_TOOLS


def test_trial_search_retains_raw_study_and_continuation(monkeypatch):
    study = {'protocolSection':{'identificationModule':{'nctId':'NCT01767311'},
                                'statusModule':{'overallStatus':'TERMINATED'}}}
    get = Mock(return_value=Mock(json=lambda:{'studies':[study],'totalCount':30,'nextPageToken':'next'}))
    monkeypatch.setattr(ct.requests,'get',get)
    result=ct.search_clinical_trials('PDAC','drug','target',['TERMINATED'])
    assert result['records']==[study] and result['truncated'] and result['next_page_token']=='next'
    params=get.call_args.kwargs['params']
    assert params['query.cond']=='PDAC' and params['filter.overallStatus']=='TERMINATED' and params['pageSize']==20


def test_paper_search_retains_all_returned_source_types_and_abstracts(monkeypatch):
    row={'id':'123','source':'MED','abstractText':'native text'}
    get=Mock(return_value=Mock(json=lambda:{'hitCount':25,'nextCursorMark':'next','resultList':{'result':[row]}}))
    monkeypatch.setattr(epmc.requests,'get',get)
    result=epmc.search_publications('PDAC AND KRAS')
    assert result['records']==[row] and result['truncated'] and result['next_cursor_mark']=='next'
    assert get.call_args.kwargs['params']=={'query':'PDAC AND KRAS','resultType':'core','format':'json','pageSize':20,'cursorMark':'*'}


@pytest.mark.parametrize('payload,status', [({'totalCount':0},'not_found'), ({},'error'),
    ({'totalCount':1,'studies':[{}]},'error')])
def test_trial_empty_or_bad_response(monkeypatch,payload,status):
    monkeypatch.setattr(ct.requests,'get',Mock(return_value=Mock(json=lambda:payload)))
    assert ct.search_clinical_trials(term='test')['status']==status


@pytest.mark.parametrize('payload,status', [({'hitCount':0,'resultList':{'result':[]}},'not_found'),
    ({},'error'),({'hitCount':1,'resultList':{'result':[{}]}},'error')])
def test_literature_empty_or_bad_response(monkeypatch,payload,status):
    monkeypatch.setattr(epmc.requests,'get',Mock(return_value=Mock(json=lambda:payload)))
    assert epmc.search_publications('test')['status']==status


def test_fulltext_identity_and_unavailable_are_not_silently_accepted(monkeypatch):
    response=Mock(status_code=200,text='<article><front><article-meta><article-id pub-id-type="pmc">123</article-id></article-meta></front></article>')
    monkeypatch.setattr(epmc.requests,'get',Mock(return_value=response))
    assert epmc.get_open_access_full_text('PMC123')['status']=='success'
    assert epmc.get_open_access_full_text('PMC124')['status']=='error'
    response.status_code=404
    assert epmc.get_open_access_full_text('PMC123')['status']=='unavailable'


def test_source_search_tools_are_registered_and_require_query():
    names={t.name for t in DISEASE_TARGET_TOOLS}
    assert {'clinicaltrials_gov_search','europe_pmc_search','get_publication_full_text_tool'}<=names
    with pytest.raises(ValueError): ct.search_clinical_trials()
    with pytest.raises(ValueError): epmc.search_publications(' ')
