# User Guide

## 1. Download

Either use GitHub **Code > Download ZIP**, or:

```powershell
git clone https://github.com/bao-kai/ncku-drug-discovery-agent1.git
cd ncku-drug-discovery-agent1
```

## 2. Create the Python environment

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
Copy-Item .env.example .env
```

Do not commit `.env`.

## 3. Run the offline test suite

```powershell
python -m pytest -q -p no:cacheprovider
```

The repository baseline is 150 passing tests. Offline tests do not require Ollama or live biomedical APIs.

## 4. Install and verify Ollama for Planner commands

Install Ollama separately, then pull the configured model:

```powershell
ollama pull qwen3:14b
ollama list
ollama run qwen3:14b "Reply with OK"
```

Keep Ollama running before a real-model Planner command.

## 5. Common commands

Show all commands:

```powershell
python agent.py --help
```

Discover disease-associated targets:

```powershell
python agent.py discover-targets --disease "Alzheimer's disease" --top 10 --output outputs
```

Collect an exact disease-target pair:

```powershell
python agent.py agent1 --disease "pancreatic cancer" --target "KRAS" --output outputs/agent1_kras.json
```

Create a plan for a known disease and optional target:

```powershell
python agent.py plan --disease "pancreatic cancer" --target "KRAS" --output outputs/plans
```

Create a plan-only Query Manifest V2:

```powershell
python agent.py plan-query --question "胰臟癌目前有哪些已知的治療標的？" --output outputs/plans/question.json
```

The Planner may take many minutes. It performs no evidence retrieval. A sibling report records attempts and diagnostics.

Experimental staged planning:

```powershell
python agent.py plan-query-staged --question "胰臟癌目前有哪些已知的治療標的？" --output outputs/plans/staged.json
```

## 6. Reading results

- A Plan file describes intended work; it is not biomedical evidence.
- A report file records Planner attempts and failures.
- Agent 1 evidence contains database facts and provenance.
- Model interpretation must remain separate from raw facts.
- Open Targets association scores are not clinical-success probabilities.
- Missing results may mean timeout, truncation, unsupported retrieval, or true absence; inspect status and limitations.

## 7. Before changing code

Read `PROJECT_STATUS.md`, `docs/AGENT1_HANDOFF.md`, `docs/ARCHITECTURE.md`, and `docs/ACCEPTANCE_CRITERIA.md`. Run the offline suite before and after each change.