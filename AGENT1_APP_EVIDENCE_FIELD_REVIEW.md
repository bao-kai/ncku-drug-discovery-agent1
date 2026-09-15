# Agent 1 APP Full Evidence Review

Source reviewed: `D:\NCKU_Drug_Discovery\outputs\agent1_APP_full_evidence.json`

## Run summary

- Disease: Alzheimer disease (`MONDO_0004975`)
- Target: APP (`ENSG00000142192`)
- Datasource expanded: `clinical_precedence`
- Aggregated datasource score: `0.9774319311654196`
- Evidence records: 143 of 143
- Truncated: false
- Possible evidence fields: 86
- Fields populated at least once: 19

## Five candidate information groups for team decision

The team should mark each group as `summary-required`, `raw-only`, or
`do-not-collect` before the Stage 1 schema is finalized.

### 1. Relationship identity

- Disease ID and source disease name
- Target ID
- Drug name or identifier
- Clinical report or study identifier
- Evidence identifier

Coverage in this run: core relationship and report identifiers were present in
143 of 143 records.

### 2. Clinical maturity

- Clinical stage
- Approval status
- Study start date
- Evidence or publication date
- Highest stage reached after records are grouped by drug

Coverage in this run: clinical stage was present in 143 records; study and
evidence dates were present in 118 records.

### 3. Outcome and risk signals

- Trial stop reason
- Stop-reason categories
- Explicit negative-result marker
- Safety or side-effect marker
- Ongoing, completed, failed, or unclear result classification when supported

Coverage in this run: 24 records contained stop information. Absence of a stop
reason must not be interpreted as a positive result.

### 4. Direction and mechanism

- Direction on target
- Direction on disease trait
- Mechanism of action and action type when available from linked sources
- Whether the proposed therapeutic direction is consistent across records

Coverage in this run: direction on trait was present in 143 records; direction
on target was present in 62 records.

### 5. Provenance and traceability

- Datasource and datatype
- Per-record evidence score
- Literature identifiers
- Original evidence ID
- Retrieval time, pagination, truncation, and collection warnings

Coverage in this run: evidence IDs and scores were present in 143 records;
literature identifiers were present in 65 records.

## Important interpretation limits

- Record count measures how much the relationship was recorded or studied; it
  does not measure efficacy.
- Clinical stage measures development maturity; it does not prove a positive
  result.
- The aggregated Open Targets score is not a probability or clinical success
  rate.
- Missing outcome fields must remain unknown and must not be inferred as
  positive or negative.
- Raw records should be retained even when the human-readable summary keeps
  only selected fields.
