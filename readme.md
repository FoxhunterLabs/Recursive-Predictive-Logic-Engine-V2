________________________________________
Recursive Predictive Logic Engine — Bayesian Edition
A deterministic, inspectable evidence engine that ingests structured operational data, measures novelty, updates per-domain reliability through online Bayesian inference, and outputs transparent accuracy and confidence metrics. Includes a full HTTP API, CSV ingestion pipeline, cycle simulator, and a real-time dashboard with historical insights and performance visualization.
________________________________________
🔍 What This Engine Does
This system processes incoming events (API, webform, CSV), evaluates their similarity to recent history, derives binary outcomes, and updates domain-specific reliability models using Bayesian Beta distributions. Each iteration generates a structured insight containing:
•	Global accuracy (posterior mean)
•	Confidence (posterior certainty)
•	Per-domain evidence counts
•	Novelty contribution
•	Delta vs. previous cycle
•	Full state snapshot for auditability
The design emphasizes determinism, replayability, and clarity — no black-box learning paths.
________________________________________
🧠 Core Concepts
Bayesian Reliability Tracking
Each domain (safety, schedule, cost) maintains:
Beta(alpha, beta)
which updates online based on:
•	Event outcomes (success/failure)
•	Novelty-weighted contributions
•	Sliding-window recency filters
The posterior mean becomes accuracy, and variance drives confidence.
Novelty Detection
Novelty = 1 − average_similarity_to_recent_entries,
computed via key/value overlap across a rolling window.
Deterministic Update Loop
Every cycle:
1.	Ingest event
2.	Compute novelty
3.	Derive outcome
4.	Update Bayesian beliefs
5.	Generate insight
6.	Apply optional feedback filters
7.	Render dashboard + API response
________________________________________
🛠️ Features
•	Real-time dashboard (Flask)
•	Bayesian domain models with transparent priors
•	Cycle runner (UI + REST API)
•	CSV ingestion
•	Historical insight log
•	Accuracy progression plot
•	Deterministic state export
________________________________________
🚀 API Endpoints
POST /api/run_cycles
Runs iterative cycles and returns insight messages plus engine state.
GET /api/state
Returns full engine snapshot:
•	Bayesian parameters
•	Latest insight
•	Recent deltas
•	Metadata totals
POST /upload
Upload a CSV file; each row becomes an event.
GET /
Interactive dashboard.
________________________________________
🗂️ Input Data Schema
Events may include:
domain: safety | schedule | cost
incident_count: int
delay_minutes: int
crew_count: int
outcome: 0/1 (optional override)
Outcome derivation rules:
•	outcome field → used directly
•	If no outcome:
o	incident_count == 0 → success
o	delay_minutes <= 15 → success
o	Otherwise → failure
________________________________________
📊 Insight Structure
Each cycle produces:
{
  "cycle": 12,
  "refined_accuracy": 0.78,
  "decision_confidence": 0.63,
  "delta": 0.04,
  "domain_stats": {
    "safety":   { "mean_accuracy": 0.80, "confidence": 0.71, "n": 42 },
    "schedule": { "mean_accuracy": 0.73, "confidence": 0.60, "n": 33 }
  },
  "novelty_contribution": 12.4,
  "volume": 55
}
________________________________________
📦 Installation
pip install flask matplotlib
python app.py
Dashboard runs at:
http://localhost:5000
________________________________________
🧪 Quick Start (API)
curl -X POST http://localhost:5000/api/run_cycles \
     -H "Content-Type: application/json" \
     -d '{"cycles": 5, "bias": 1.0}'
________________________________________
🔒 Design Principles
•	Deterministic, inspectable state transitions
•	No hidden training, no nondeterministic randomness
•	Evidence-driven, not heuristic-driven
•	Fully auditable with reproducible results
•	Lightweight enough for edge devices or embedded autonomy nodes
________________________________________
📘 License
MIT License.
________________________________________
