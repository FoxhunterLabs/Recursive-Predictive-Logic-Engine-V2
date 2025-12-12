import json
import uuid
import time
import argparse
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, render_template_string
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

metadata_bank: List[Dict[str, Any]] = []
novel_insights: List[Dict[str, Any]] = []
history_log: List[Dict[str, Any]] = []

engine_state = {
    'baseline_accuracy': 0.60,
    'confidence_multiplier': 1.0,
    'refinement_rate': 0.08,  # legacy knob, still tracked but not core to Bayes
    'loop_count': 0,
    'weight_bias': 1.0,
    'domain_counts': {
        'safety': 0,
        'schedule': 0,
        'cost': 0
    },
    'novelty_threshold': 0.2,
    'decay_days': 7,
    # Bayesian α/β per domain (prior mean ~0.6 with α=6, β=4)
    'domain_betas': {
        'safety':   {'alpha': 6.0, 'beta': 4.0},
        'schedule': {'alpha': 6.0, 'beta': 4.0},
        'cost':     {'alpha': 6.0, 'beta': 4.0},
    }
}


def log(event: str) -> None:
    entry = {'timestamp': datetime.now().isoformat(), 'event': event}
    history_log.append(entry)
    print(f"[LOG] {entry['timestamp']}: {event}")


def calculate_novelty(entry: Dict[str, Any]) -> float:
    """
    Novelty = 1 - average_similarity_to_recent

    Similarity between two entries:
        matches / union_of_keys
    """
    recent = metadata_bank[-20:]
    if not recent:
        return 1.0

    similarity_scores: List[float] = []

    for old in recent:
        matches = sum(
            1 for k in entry
            if k in old and entry[k] == old[k]
        )
        total_possible = len(set(entry.keys()) | set(old.keys()))
        similarity_scores.append(
            matches / total_possible if total_possible else 1.0
        )

    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    return round(max(0.0, 1.0 - avg_similarity), 3)


def derive_outcome(entry: Dict[str, Any]) -> Optional[int]:
    """
    Derive a binary outcome from the entry.
    Priority:
      1. Explicit 'outcome' field
      2. 'incident_count' (0 = success, >0 = failure)
      3. 'delay_minutes' (<= 15 = success, > 15 = failure)
    Returns 1 (success), 0 (failure), or None if cannot infer.
    """
    if 'outcome' in entry:
        raw = entry.get('outcome')
        if isinstance(raw, bool):
            return 1 if raw else 0
        if isinstance(raw, (int, float)):
            return 1 if raw >= 0.5 else 0
        if isinstance(raw, str):
            v = raw.strip().lower()
            if v in {'1', 'true', 'yes', 'y'}:
                return 1
            if v in {'0', 'false', 'no', 'n'}:
                return 0
            try:
                num = float(v)
                return 1 if num >= 0.5 else 0
            except ValueError:
                return None
        return None

    if 'incident_count' in entry:
        try:
            incidents = int(entry['incident_count'])
            return 1 if incidents == 0 else 0
        except (ValueError, TypeError):
            pass

    if 'delay_minutes' in entry:
        try:
            delay = float(entry['delay_minutes'])
            return 1 if delay <= 15 else 0
        except (ValueError, TypeError):
            pass

    return None


def update_domain_belief(entry: Dict[str, Any]) -> None:
    """
    Update Bayesian α/β for the given domain based on the binary outcome.
    Novelty acts as a fractional weight (0.1–1.0) on the update size.
    """
    domain = entry.get('domain')
    if domain not in engine_state['domain_betas']:
        return

    outcome = entry.get('outcome')
    if outcome is None:
        return

    outcome = 1 if outcome else 0

    novelty = entry.get('novelty_score', 1.0)
    weight = max(0.1, min(1.0, float(novelty)))

    beta_params = engine_state['domain_betas'][domain]
    if outcome == 1:
        beta_params['alpha'] += weight
    else:
        beta_params['beta'] += weight


def ingest_data(payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload['id'] = str(uuid.uuid4())
    payload['timestamp'] = datetime.now().isoformat()

    domain = payload.get('domain')
    if domain in engine_state['domain_counts']:
        engine_state['domain_counts'][domain] += 1

    payload['outcome'] = derive_outcome(payload)
    payload['novelty_score'] = calculate_novelty(payload)

    metadata_bank.append(payload)
    update_domain_belief(payload)

    log(
        f"Data ingested: {payload['id']} | Domain: {domain} | "
        f"Outcome: {payload['outcome']} | Novelty: {payload['novelty_score']:.3f}"
    )


def refine_predictions() -> Dict[str, Any]:
    """
    Compute global and per-domain accuracy and confidence
    from Bayesian α/β, and summarize recent novelty.
    """
    engine_state['loop_count'] += 1
    now = datetime.now()

    recent_data = [
        m for m in metadata_bank
        if datetime.fromisoformat(m['timestamp']) >
        now - timedelta(days=engine_state['decay_days'])
    ]
    volume = len(recent_data)
    novelty_values = [m.get('novelty_score', 1.0) for m in recent_data]
    novelty_contribution = sum(novelty_values) if novelty_values else 0.0

    domain_stats: Dict[str, Dict[str, Any]] = {}
    combined_alpha = 0.0
    combined_beta = 0.0

    for domain, params in engine_state['domain_betas'].items():
        alpha = float(params['alpha'])
        beta = float(params['beta'])
        n = alpha + beta

        if n <= 0:
            mean = engine_state['baseline_accuracy']
            variance = 0.25
        else:
            mean = alpha / n
            variance = (alpha * beta) / (n ** 2 * (n + 1))

        confidence = max(0.0, min(1.0, 1.0 - variance * 10))

        domain_stats[domain] = {
            'mean_accuracy': round(mean, 3),
            'confidence': round(confidence, 3),
            'n': round(n, 1)
        }

        combined_alpha += alpha
        combined_beta += beta

    total_n = combined_alpha + combined_beta
    if total_n > 0:
        global_mean = combined_alpha / total_n
    else:
        global_mean = engine_state['baseline_accuracy']

    refined_accuracy = round(min(0.98, max(0.02, global_mean)), 3)

    global_conf = round(
        min(1.0, (total_n / (total_n + 20.0))) *
        engine_state['confidence_multiplier'], 3
    )

    delta = 0.0
    if novel_insights:
        delta = round(refined_accuracy - novel_insights[-1]['refined_accuracy'], 3)

    insight = {
        'cycle': engine_state['loop_count'],
        'volume': volume,
        'refined_accuracy': refined_accuracy,
        'decision_confidence': global_conf,
        'delta': delta,
        'domain_breakdown': {d: s['n'] for d, s in domain_stats.items()},
        'domain_stats': domain_stats,
        'novelty_contribution': round(novelty_contribution, 3),
        'timestamp': datetime.now().isoformat()
    }
    novel_insights.append(insight)
    log(f"Insight generated: {insight}")
    return insight


def apply_filters_and_feedback() -> None:
    """
    Legacy hook to adjust pacing and inject feedback.
    Kept minimal to avoid corrupting Bayesian state.
    """
    if len(novel_insights) < 2:
        return

    latest = novel_insights[-1]

    if latest['refined_accuracy'] > 0.95:
        engine_state['refinement_rate'] *= 0.95
        log("Tapering refinement rate due to high global accuracy.")

    metadata_bank.append({
        'feedback': f"Loop {latest['cycle']} feedback",
        'accuracy': latest['refined_accuracy'],
        'confidence': latest['decision_confidence'],
        'loop_count': latest['cycle'],
        'novelty_score': 0.5,
        'timestamp': datetime.now().isoformat()
    })
    log("Filtered feedback injected into metadata_bank.")


def generate_plot() -> str:
    if not novel_insights:
        return ""
    x = [insight['cycle'] for insight in novel_insights]
    y = [insight['refined_accuracy'] for insight in novel_insights]
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title('Refined Accuracy Over Time')
    plt.xlabel('Cycle')
    plt.ylabel('Accuracy')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return f"<img src='data:image/png;base64,{img_data}' />"


def generate_insight_message(insight: Dict[str, Any]) -> str:
    messages: List[str] = []

    if insight['delta'] > 0.02:
        messages.append(f"Global accuracy improved by {insight['delta'] * 100:.1f}%")
    elif insight['delta'] < -0.02:
        messages.append(f"Global accuracy dropped by {abs(insight['delta']) * 100:.1f}%")

    domain_stats = insight.get('domain_stats', {})
    if domain_stats:
        max_domain = max(domain_stats, key=lambda d: domain_stats[d]['n'])
        ds = domain_stats[max_domain]
        messages.append(
            f"{max_domain.capitalize()} has {ds['n']:.0f} labeled events, "
            f"{ds['mean_accuracy'] * 100:.1f}% accuracy, "
            f"{ds['confidence'] * 100:.0f}% confidence"
        )

    if insight.get('novelty_contribution', 0) > 0 and insight.get('volume', 0) > 0:
        avg_novelty = insight['novelty_contribution'] / max(insight['volume'], 1)
        messages.append(f"Average novelty of recent data: {avg_novelty:.3f}")

    if not messages:
        messages.append(
            f"Cycle {insight['cycle']} completed with "
            f"{insight['refined_accuracy'] * 100:.1f}% global accuracy "
            f"at {insight['decision_confidence'] * 100:.0f}% confidence"
        )

    return "; ".join(messages)


def export_state() -> Dict[str, Any]:
    return {
        'engine': engine_state,
        'latest_insight': novel_insights[-1] if novel_insights else {},
        'total_metadata': len(metadata_bank),
        'history': history_log[-5:],
        'deltas': [i['delta'] for i in novel_insights[-5:]]
    }


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'v2.1-bayes-novelty'}), 200


@app.route('/')
def dashboard():
    state = export_state()
    graph_html = generate_plot()
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recursive Predictive Logic Engine</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                h2 { color: #333; }
                h3 { color: #555; margin-top: 20px; }
                pre { background: #fff; padding: 15px; border-radius: 5px; overflow-x: auto; }
                form { background: #fff; padding: 15px; margin: 15px 0; border-radius: 5px; }
                label { display: block; margin: 10px 0 5px; font-weight: bold; }
                input[type="number"], input[type="file"] { padding: 8px; width: 200px; }
                input[type="range"] { width: 200px; }
                button { background: #007bff; color: white; padding: 10px 20px;
                         border: none; border-radius: 5px; cursor: pointer; margin-top: 10px; }
                button:hover { background: #0056b3; }
                button:disabled { background: #6c757d; cursor: not-allowed; }
                hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
                img { max-width: 100%; border-radius: 5px; margin: 15px 0; }
                #insights { background: #fff; padding: 15px; border-radius: 5px;
                            margin: 15px 0; min-height: 100px; }
                .insight-card { background: #f8f9fa; padding: 12px; margin: 10px 0;
                                border-left: 4px solid #007bff; border-radius: 4px; }
                .insight-card b { color: #333; display: block; margin-bottom: 5px; }
                .insight-meta { color: #666; font-size: 0.9em; }
                .loading { color: #007bff; font-style: italic; }
                .api-section { background: #fff; padding: 15px; border-radius: 5px; margin: 15px 0; }
            </style>
        </head>
        <body>
            <h2>Recursive Predictive Logic Engine — v2.1 (Bayesian + Correct Novelty)</h2>
            <pre id="state">{{ state|tojson(indent=2) }}</pre>
            
            <div class="api-section">
                <h3>Dynamic Cycle Execution</h3>
                <label>Number of cycles:</label>
                <input type="number" id="api-cycles" value="3" min="1" max="20">
                <label>Weight bias (1.0 = neutral):</label>
                <input type="range" id="api-bias" min="0.1" max="2.0" step="0.1" value="1.0"
                       oninput="document.getElementById('bias-value').textContent = this.value">
                <output id="bias-value">1.0</output>
                <button id="api-run-btn" onclick="runCyclesAPI()">Run Cycles (API)</button>
            </div>
            
            <div id="insights">
                <h3>Insights</h3>
                <p style="color: #999;">Run cycles to generate insights...</p>
            </div>
            
            <hr>
            
            <form action="/run" method="post">
                <label>Number of cycles:</label>
                <input type="number" name="cycles" value="3" min="1" max="20">
                <label>Weight bias (1.0 = neutral):</label>
                <input type="range" name="bias" min="0.1" max="2.0" step="0.1" value="1.0"
                       oninput="this.nextElementSibling.value = this.value">
                <output>1.0</output>
                <button type="submit">Run Cycles (Full Refresh)</button>
            </form>
            <hr>
            {{ graph|safe }}
            <form action="/upload" method="post" enctype="multipart/form-data">
                <label>Upload CSV:</label>
                <input type="file" name="file" accept=".csv">
                <button type="submit">Upload</button>
            </form>
            
            <script>
                function runCyclesAPI() {
                    const cycles = parseInt(document.getElementById('api-cycles').value);
                    const bias = parseFloat(document.getElementById('api-bias').value);
                    const btn = document.getElementById('api-run-btn');
                    const insightsBox = document.getElementById('insights');
                    
                    btn.disabled = true;
                    btn.textContent = 'Running...';
                    insightsBox.innerHTML = '<h3>Insights</h3><p class="loading">Processing cycles...</p>';
                    
                    fetch('/api/run_cycles', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ cycles: cycles, bias: bias })
                    })
                    .then(r => r.json())
                    .then(data => {
                        if (data.status === 'success') {
                            insightsBox.innerHTML = '<h3>Insights</h3>' + 
                                data.insights.map(i => 
                                    `<div class="insight-card">
                                        <b>${i.insight}</b>
                                        <div class="insight-meta">
                                            Confidence: ${i.confidence} | 
                                            Novelty: ${i.novelty_score} | 
                                            Accuracy: ${(i.accuracy * 100).toFixed(1)}% |
                                            Cycle: ${i.cycle}
                                        </div>
                                    </div>`
                                ).join('');
                            
                            document.getElementById('state').textContent = 
                                JSON.stringify(data.engine_state, null, 2);
                        }
                        btn.disabled = false;
                        btn.textContent = 'Run Cycles (API)';
                    })
                    .catch(err => {
                        console.error('Error:', err);
                        insightsBox.innerHTML =
                            '<h3>Insights</h3><p style="color: red;">Error running cycles</p>';
                        btn.disabled = false;
                        btn.textContent = 'Run Cycles (API)';
                    });
                }
            </script>
        </body>
        </html>
    ''',
                                  state=state,
                                  graph=graph_html)


@app.route('/run', methods=['POST'])
def run_cycles():
    try:
        cycles = int(request.form.get('cycles', 3))
        bias = float(request.form.get('bias', 1.0))
    except (ValueError, TypeError):
        return "Invalid input: cycles and bias must be valid numbers", 400
    
    if not (1 <= cycles <= 20):
        return "Error: cycles must be between 1 and 20", 400
    
    if not (0.1 <= bias <= 2.0):
        return "Error: bias must be between 0.1 and 2.0", 400
    
    engine_state['weight_bias'] = bias
    for i in range(cycles):
        ingest_data({
            'source': 'webform',
            'incident_count': i,
            'delay_minutes': i * 3,
            'crew_count': 5 + i,
            'domain': 'safety' if i % 2 == 0 else 'schedule'
        })
        refine_predictions()
        apply_filters_and_feedback()
        time.sleep(0.1)
    return dashboard()


@app.route('/upload', methods=['POST'])
def upload_csv():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.DictReader(stream)
    for row in csv_input:
        ingest_data(row)
        refine_predictions()
        apply_filters_and_feedback()
    return dashboard()


@app.route('/api/run_cycles', methods=['POST'])
def api_run_cycles():
    data = request.get_json() or {}
    
    try:
        cycles_raw = data.get('cycles', 3)
        bias = float(data.get('bias', 1.0))
        
        if isinstance(cycles_raw, float) and not cycles_raw.is_integer():
            return jsonify({
                'status': 'error',
                'message': 'cycles must be an integer, not a decimal number'
            }), 400
        
        cycles = int(cycles_raw)
    except (ValueError, TypeError):
        return jsonify({
            'status': 'error',
            'message': 'Invalid input: cycles must be an integer and bias must be a number'
        }), 400
    
    if not (1 <= cycles <= 20):
        return jsonify({
            'status': 'error',
            'message': 'cycles must be between 1 and 20'
        }), 400
    
    if not (0.1 <= bias <= 2.0):
        return jsonify({
            'status': 'error',
            'message': 'bias must be between 0.1 and 2.0'
        }), 400
    
    engine_state['weight_bias'] = bias
    insights: List[Dict[str, Any]] = []
    
    for i in range(cycles):
        ingest_data({
            'source': 'api',
            'incident_count': i,
            'delay_minutes': i * 3,
            'crew_count': 5 + i,
            'domain': 'safety' if i % 2 == 0 else 'schedule'
        })
        insight = refine_predictions()
        apply_filters_and_feedback()
        
        if insight['volume'] > 0:
            avg_novelty = round(
                insight['novelty_contribution'] / max(insight['volume'], 1), 3
            )
        else:
            avg_novelty = 0.0
        
        formatted_insight = {
            'insight': generate_insight_message(insight),
            'confidence': insight['decision_confidence'],
            'novelty_score': avg_novelty,
            'cycle': insight['cycle'],
            'accuracy': insight['refined_accuracy']
        }
        insights.append(formatted_insight)
        time.sleep(0.05)
    
    return jsonify({
        'status': 'success',
        'insights': insights,
        'engine_state': export_state()
    })


@app.route('/api/state', methods=['GET'])
def api_get_state():
    return jsonify(export_state())


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0',
            port=5000,
            debug=os.environ.get('FLASK_ENV') != 'production')
