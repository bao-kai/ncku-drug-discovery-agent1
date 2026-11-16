"""Single-study ClinicalTrials.gov lookup for Agent 1 supplemental retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

import requests
from pydantic import BaseModel


ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"
TIMEOUT_SECONDS = 30


def search_clinical_trials(condition: str = '', intervention: str = '', term: str = '',
                           overall_statuses: list[str] | None = None,
                           page_token: str | None = None) -> dict:
    """Fetch one bounded page of native studies; keywords do not prove target engagement."""
    if not any(v.strip() for v in (condition, intervention, term)):
        raise ValueError('Supply a condition, intervention or search term')
    params = {'format': 'json', 'pageSize': 20, 'countTotal': 'true'}
    for key, value in [('query.cond', condition), ('query.intr', intervention), ('query.term', term)]:
        if value.strip():
            params[key] = value.strip()
    if overall_statuses:
        params['filter.overallStatus'] = ','.join(overall_statuses)
    if page_token:
        params['pageToken'] = page_token
    result = {'source': 'clinicaltrials_gov', 'query_parameters': params,
              'source_url': requests.Request('GET', ENDPOINT, params=params).prepare().url,
              'retrieved_at': datetime.now(timezone.utc).isoformat(), 'status': 'error',
              'records': [], 'total_count': None, 'next_page_token': None,
              'truncated': None, 'error': None}
    try:
        response = requests.get(ENDPOINT, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        total = int(payload['totalCount'])
        rows = payload.get('studies', [])
        if not isinstance(rows, list) or len(rows) > 20 or (total > 0 and not rows and not page_token):
            raise ValueError('Malformed study search response')
        ids = [normalize_nct_id(row['protocolSection']['identificationModule']['nctId']) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate study IDs in page')
        next_page = payload.get('nextPageToken')
        result.update(status='success' if rows else 'not_found', records=rows,
                      total_count=total, next_page_token=next_page,
                      truncated=bool(next_page) or (not page_token and total > len(rows)))
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


class ClinicalTrialLookup(BaseModel):
    source: Literal["clinicaltrials_gov"] = "clinicaltrials_gov"
    record_id: str
    source_url: str
    retrieved_at: datetime
    status: Literal["success", "not_found", "error"]
    raw_record: dict[str, Any] | None = None
    error: str | None = None


def normalize_nct_id(nct_id: str) -> str:
    normalized = nct_id.strip().upper()
    if not re.fullmatch(r"NCT[0-9]{8}", normalized):
        raise ValueError("Expected a ClinicalTrials.gov ID such as NCT04241068")
    return normalized


def get_clinical_trial(nct_id: str) -> ClinicalTrialLookup:
    """Fetch one exact ID, preserving the full response and explicit failures."""
    nct_id = normalize_nct_id(nct_id)
    url = f"{ENDPOINT}/{nct_id}"
    metadata = {
        "record_id": nct_id,
        "source_url": url,
        "retrieved_at": datetime.now(timezone.utc),
    }
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        if response.status_code == 404:
            return ClinicalTrialLookup(**metadata, status="not_found")
        response.raise_for_status()
        raw = response.json()
        actual_id = raw["protocolSection"]["identificationModule"]["nctId"]
        if actual_id != nct_id:
            raise ValueError(f"Requested {nct_id}, received {actual_id}")
        return ClinicalTrialLookup(**metadata, status="success", raw_record=raw)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return ClinicalTrialLookup(
            **metadata, status="error", error=f"{type(exc).__name__}: {exc}"
        )
