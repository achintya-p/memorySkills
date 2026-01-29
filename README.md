# Skills + Memory Tester: Evaluation Harness for Agents with Memory + Skills


**Features:**
- ✅ Componentized architecture (MemoryStore / SkillRegistry / AgentPolicy)
- ✅ Reproducible episode runner + full execution traces
- ✅ Benign capability tracks + robustness/poisoning tracks
- ✅ Metrics that attribute failures to write vs retrieve vs apply
- ✅ Multiple memory backends (List, KV, extensible)
- ✅ Heuristic + LLM-ready skill routing
- ✅ Automated report generation (JSON + HTML)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run full evaluation
python src/orchestrator.py --full

# View results
open evaluationData/report_*.html
```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage.

---

## Architecture Overview

### Core Components

#### 1. **MemoryStore** (src/memory_store.py)
Pluggable memory backend with:
- **ListMemoryStore**: Recency-based, substring matching (baseline)
- **KVMemoryStore**: Keyed, latest-write-wins (production-like)
- 6 namespace types: episodic, semantic, preferences, tool_traces, skills, working
- Eviction policies: LRU, oldest-first, TTL
- Full operation logging

#### 2. **SkillRegistry** (src/skill_registry.py)
YAML-defined skill management with heuristic router, safety validation, and decision logging.

#### 3. **AgentPolicy** (src/agent_policy.py)
Central decision engine: memory selection, skill routing, prompt building. Principle: memory/skills are "untrusted hints".

#### 4. **EpisodeRunner** (src/episode_runner.py)
Standardized episode execution with reproducible traces and post-hoc failure attribution.

#### 5. **Evaluation Tracks** (src/evaluation_tracks.py)
10+ episodes: benign capability (preference recall, procedural continuity, tool reuse, provenance) + robustness (R1 poisoning, R2 persistence).

#### 6. **Metrics + Attribution** (src/metrics.py)
Task completion, consistency, skill accuracy, attack success rate, poison metrics, and failure attribution.

#### 7. **Evaluation Reporter** (src/evaluation_reporter.py)
Publishable reports with executive summary, per-track breakdowns, HTML + JSON output.

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.

---

## Installation & Usage

```bash
pip install -r requirements.txt

# Full evaluation
python src/orchestrator.py --full

# Custom backend
python src/orchestrator.py --full --memory-backend kv

# Custom skills
python src/orchestrator.py --full --skills-yaml /path/to/skills.yaml
```

---

## Evaluation Tracks

**Benign Capability:**
- Preference recall (long horizon, updates/conflicts)
- Procedural continuity (multi-step workflows)
- Tool reuse (pattern memorization)
- Provenance tracking (fact attribution)

**Robustness:**
- R1: Write-surface, retrieval, procedural poisoning
- R2: Delayed triggers, policy injection, flooding attacks

---

## Output

After running, check:
- `evaluationData/report_*.html` – Visual report (open in browser)
- `evaluationData/report_*.json` – Machine-readable metrics
- `evaluationData/traces_*.jsonl` – Detailed execution logs

---

## File Structure

```
skills:memorytester/
├── README.md                    # This file
├── ARCHITECTURE.md              # Technical architecture
├── QUICKSTART.md                # Quick start guide
├── requirements.txt
├── skills.yaml                  # Skill definitions
├── src/
│   ├── memory_store.py          # Memory backends
│   ├── skill_registry.py        # Skill management
│   ├── agent_policy.py          # Decision engine
│   ├── episode_runner.py        # Episode execution
│   ├── evaluation_tracks.py     # Test scenarios
│   ├── metrics.py               # Metrics computation
│   ├── evaluation_reporter.py   # Report generation
│   └── orchestrator.py          # Main orchestrator
├── evaluationData/              # Generated results
└── testCases/                   # Legacy test data
```

---

## Extending the Framework

### Add Custom Memory Backend
See `src/memory_store.py` for `MemoryStoreBase` abstract class.

### Add Custom Skills
Edit `skills.yaml` with new skill definitions.

### Add Custom Episodes
Extend `src/evaluation_tracks.py` with new episode functions.

---

## Metrics Reference

| Track | Metric | Range | Ideal |
|-------|--------|-------|-------|
| Benign | Task Completion | 0-1 | ≥0.9 |
| Benign | Consistency | 0-1 | ≥0.95 |
| Robustness | Attack Success Rate | 0-1 | 0 |
| Robustness | Poison Utilization | 0-1 | 0 |

**Attribution:** Write vs Retrieve vs Apply faults

---

## Defenses Implemented

- **Write-time**: Storeworthiness filter (source, confidence checks)
- **Read-time**: Instruction stripping, trust-score filtering
- **System-level**: Explicit "untrusted hints" framing in prompts

---

## Requirements

- Python 3.8+
- Dependencies: requests, openai, python-dotenv, pyyaml

---

## For More Information

- [QUICKSTART.md](QUICKSTART.md) – Setup and usage guide
- [ARCHITECTURE.md](ARCHITECTURE.md) – Technical deep dive