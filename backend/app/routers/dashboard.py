from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """
    EvalForge browser dashboard.

    This frontend is intentionally lightweight: FastAPI serves a single
    HTML/JS page that talks to the existing backend endpoints.
    """

    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>EvalForge Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e5e7eb;
      --text: #111827;
      --muted: #6b7280;
      --blue: #2563eb;
      --blue-dark: #1d4ed8;
      --green: #059669;
      --red: #dc2626;
      --yellow: #d97706;
      --purple: #7c3aed;
      --shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
      --radius: 18px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      background: linear-gradient(135deg, #0f172a, #1e3a8a);
      color: white;
      padding: 34px 38px;
    }

    header h1 {
      margin: 0;
      font-size: 36px;
      letter-spacing: -0.03em;
    }

    header p {
      margin: 10px 0 0 0;
      max-width: 920px;
      color: #dbeafe;
      line-height: 1.6;
    }

    main {
      max-width: 1280px;
      margin: 28px auto 60px auto;
      padding: 0 24px;
    }

    .grid {
      display: grid;
      gap: 20px;
    }

    .grid-2 {
      grid-template-columns: 1.05fr 0.95fr;
    }

    .grid-4 {
      grid-template-columns: repeat(4, 1fr);
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px;
      box-shadow: var(--shadow);
    }

    .card h2 {
      margin: 0 0 14px 0;
      font-size: 22px;
      letter-spacing: -0.02em;
    }

    .card h3 {
      margin: 0 0 10px 0;
      font-size: 17px;
    }

    .muted {
      color: var(--muted);
      line-height: 1.5;
    }

    label {
      display: block;
      margin-top: 14px;
      margin-bottom: 6px;
      font-weight: 650;
      font-size: 14px;
    }

    input, select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      font-size: 15px;
      background: white;
    }

    input[type="file"] {
      background: #f3f4f6;
      border-style: dashed;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    button {
      border: 0;
      border-radius: 12px;
      padding: 12px 16px;
      background: var(--blue);
      color: white;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      transition: 0.15s ease;
    }

    button:hover {
      background: var(--blue-dark);
      transform: translateY(-1px);
    }

    button.secondary {
      background: #111827;
    }

    button.light {
      background: #e5e7eb;
      color: #111827;
    }

    button.green {
      background: var(--green);
    }

    button.red {
      background: var(--red);
    }

    .button-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }

    .metric-card {
      background: white;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .metric-value {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 9px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
      margin-right: 6px;
      margin-top: 4px;
    }

    .pill.green {
      background: #ecfdf5;
      color: #047857;
    }

    .pill.red {
      background: #fef2f2;
      color: #b91c1c;
    }

    .pill.yellow {
      background: #fffbeb;
      color: #b45309;
    }

    .case-card {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-top: 12px;
      background: #ffffff;
    }

    .case-query {
      font-weight: 750;
      margin-bottom: 8px;
    }

    pre {
      background: #0f172a;
      color: #e5e7eb;
      padding: 14px;
      border-radius: 14px;
      overflow-x: auto;
      max-height: 420px;
      font-size: 13px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    th, td {
      text-align: left;
      padding: 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }

    th {
      color: #374151;
      background: #f9fafb;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .status {
      padding: 12px 14px;
      border-radius: 12px;
      margin-top: 14px;
      display: none;
    }

    .status.ok {
      display: block;
      background: #ecfdf5;
      color: #065f46;
      border: 1px solid #a7f3d0;
    }

    .status.err {
      display: block;
      background: #fef2f2;
      color: #991b1b;
      border: 1px solid #fecaca;
    }

    .status.info {
      display: block;
      background: #eff6ff;
      color: #1e40af;
      border: 1px solid #bfdbfe;
    }

    .section-title {
      margin: 30px 0 14px 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }

    .small {
      font-size: 13px;
    }

    .downloads a {
      display: inline-flex;
      margin: 6px 8px 0 0;
      color: white;
      background: #111827;
      padding: 9px 12px;
      border-radius: 10px;
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
    }

    .compare-good {
      color: var(--green);
      font-weight: 800;
    }

    .compare-bad {
      color: var(--red);
      font-weight: 800;
    }

    @media (max-width: 980px) {
      .grid-2, .grid-4, .row {
        grid-template-columns: 1fr;
      }

      header {
        padding: 28px 22px;
      }

      header h1 {
        font-size: 30px;
      }
    }
  </style>
</head>

<body>
  <header>
    <h1>EvalForge Dashboard</h1>
    <p>
      Upload RAG documents, policy files, PDFs, CSVs, and optional tool schemas.
      EvalForge automatically generates benchmark cases, validates citations,
      runs evaluation, stores results in SQLite, exports datasets, and compares runs.
    </p>
  </header>

  <main>
    <div class="grid grid-2">
      <section class="card">
        <h2>1. Run EvalForge Pipeline</h2>
        <p class="muted">
          Add your source files once. The backend handles ingestion, chunking, rule extraction,
          case generation, validation, optional evaluation, and persistence.
        </p>

        <form id="pipelineForm">
          <label>Upload Files</label>
          <input
            id="files"
            name="files"
            type="file"
            multiple
            required
            accept=".md,.txt,.json,.csv,.pdf"
          />
          <p class="muted small">
            Supported: .md, .txt, .json, .csv, .pdf. Include a tool schema JSON to generate tool-use cases.
          </p>

          <div class="row">
            <div>
              <label>Project ID</label>
              <input id="project_id" name="project_id" value="support_demo_dashboard" />
            </div>
            <div>
              <label>Dataset Version</label>
              <input id="dataset_version" name="dataset_version" value="v0.1.0" />
            </div>
          </div>

          <div class="row">
            <div>
              <label>Max Cases Per Type</label>
              <input id="max_cases_per_type" name="max_cases_per_type" type="number" min="1" max="50" value="5" />
            </div>
            <div>
              <label>Citation Support Threshold</label>
              <input id="citation_support_threshold" name="citation_support_threshold" type="number" min="0" max="1" step="0.05" value="0.35" />
            </div>
          </div>

          <div class="row">
            <div>
              <label>Run Evaluation</label>
              <select id="run_eval" name="run_eval">
                <option value="true" selected>true</option>
                <option value="false">false</option>
              </select>
            </div>
            <div>
              <label>Target System</label>
              <select id="target_system" name="target_system">
                <option value="demo_target_system" selected>demo_target_system</option>
                <option value="intentionally_bad_target_system">intentionally_bad_target_system</option>
              </select>
            </div>
          </div>

          <label>Pass Threshold</label>
          <input id="pass_threshold" name="pass_threshold" type="number" min="0" max="1" step="0.05" value="0.70" />

          <button type="submit">Run Pipeline</button>
        </form>

        <div id="status" class="status"></div>
      </section>

      <section class="card">
        <h2>2. Latest Result</h2>
        <p class="muted">
          After a run completes, this panel shows pipeline metrics, quality metrics,
          evaluation scores, and download links.
        </p>

        <div id="latestRunText" class="muted">
          No run yet. Upload files and click <strong>Run Pipeline</strong>.
        </div>

        <div id="downloadLinks" class="downloads"></div>
      </section>
    </div>

    <h2 class="section-title">Pipeline Metrics</h2>
    <section class="grid grid-4" id="metricGrid">
      <div class="metric-card">
        <div class="metric-label">Documents</div>
        <div class="metric-value" id="metricDocuments">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Chunks</div>
        <div class="metric-value" id="metricChunks">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Rules</div>
        <div class="metric-value" id="metricRules">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Cases</div>
        <div class="metric-value" id="metricCases">—</div>
      </div>
    </section>

    <h2 class="section-title">Quality and Evaluation</h2>
    <section class="grid grid-4">
      <div class="metric-card">
        <div class="metric-label">Validity Rate</div>
        <div class="metric-value" id="metricValidity">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Citation Coverage</div>
        <div class="metric-value" id="metricCitation">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Pass Rate</div>
        <div class="metric-value" id="metricPassRate">—</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Average Score</div>
        <div class="metric-value" id="metricAvgScore">—</div>
      </div>
    </section>

    <div class="grid grid-2" style="margin-top: 24px;">
      <section class="card">
        <h2>Test Type Distribution</h2>
        <div id="testTypeDistribution" class="muted">No data yet.</div>
      </section>

      <section class="card">
        <h2>Risk Distribution</h2>
        <div id="riskDistribution" class="muted">No data yet.</div>
      </section>
    </div>

    <section class="card" style="margin-top: 24px;">
      <h2>Generated Cases Preview</h2>
      <p class="muted">
        Shows the first few persisted cases from the latest pipeline run.
      </p>
      <div id="casesPreview" class="muted">No cases yet.</div>
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>Persisted Runs</h2>
      <p class="muted">
        These are saved in SQLite. You can open exports, inspect cases, or compare two runs.
      </p>
      <div class="button-row">
        <button class="light" onclick="loadRuns()">Refresh Runs</button>
      </div>
      <div id="runsTable" style="margin-top: 16px;" class="muted">No runs loaded yet.</div>
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>Run Comparison</h2>
      <p class="muted">
        Compare a current run against a baseline run to detect improvements or regressions.
      </p>

      <div class="row">
        <div>
          <label>Current Run ID</label>
          <input id="currentRunId" placeholder="pipeline_current..." />
        </div>
        <div>
          <label>Baseline Run ID</label>
          <input id="baselineRunId" placeholder="pipeline_baseline..." />
        </div>
      </div>

      <button style="margin-top: 16px;" onclick="compareRuns()">Compare Runs</button>

      <div id="compareResult" style="margin-top: 16px;" class="muted">No comparison yet.</div>
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>Raw Latest Response</h2>
      <pre id="rawOutput">No response yet.</pre>
    </section>
  </main>

  <script>
    let latestRunId = null;

    function setStatus(message, type = "info") {
      const el = document.getElementById("status");
      el.className = "status " + type;
      el.innerHTML = message;
    }

    function formatNumber(value) {
      if (value === null || value === undefined) return "—";
      if (typeof value === "number") {
        if (value <= 1 && value >= 0) return value.toFixed(3);
        return String(value);
      }
      return String(value);
    }

    function renderDistribution(containerId, distribution) {
      const el = document.getElementById(containerId);

      if (!distribution || Object.keys(distribution).length === 0) {
        el.innerHTML = "No data.";
        return;
      }

      el.innerHTML = Object.entries(distribution)
        .map(([key, value]) => `<span class="pill">${key}: ${value}</span>`)
        .join("");
    }

    function updateMetric(id, value) {
      document.getElementById(id).textContent = formatNumber(value);
    }

    function updateDashboard(data) {
      latestRunId = data.pipeline_run_id;

      updateMetric("metricDocuments", data.document_count);
      updateMetric("metricChunks", data.chunk_count);
      updateMetric("metricRules", data.rule_count);
      updateMetric("metricCases", data.case_count);

      const quality = data.quality_summary || {};
      const evalSummary = data.eval_summary || {};

      updateMetric("metricValidity", quality.validity_rate);
      updateMetric("metricCitation", quality.citation_coverage);
      updateMetric("metricPassRate", evalSummary.pass_rate);
      updateMetric("metricAvgScore", evalSummary.average_score);

      renderDistribution("testTypeDistribution", quality.test_type_distribution);
      renderDistribution("riskDistribution", quality.risk_distribution);

      document.getElementById("latestRunText").innerHTML = `
        <p><strong>Pipeline Run ID:</strong> <code>${data.pipeline_run_id || "N/A"}</code></p>
        <p><strong>Project:</strong> ${data.project_id}</p>
        <p><strong>Dataset Version:</strong> ${data.dataset_version}</p>
        <p><strong>Message:</strong> ${data.message}</p>
      `;

      if (data.pipeline_run_id) {
        document.getElementById("downloadLinks").innerHTML = `
          <a href="/pipeline/runs/${data.pipeline_run_id}/export/json" target="_blank">JSON</a>
          <a href="/pipeline/runs/${data.pipeline_run_id}/export/jsonl" target="_blank">JSONL</a>
          <a href="/pipeline/runs/${data.pipeline_run_id}/export/csv" target="_blank">CSV</a>
          <a href="/pipeline/runs/${data.pipeline_run_id}/quality-report" target="_blank">Quality Report</a>
        `;

        document.getElementById("currentRunId").value = data.pipeline_run_id;
        loadCasesPreview(data.pipeline_run_id);
      }

      document.getElementById("rawOutput").textContent = JSON.stringify(data, null, 2);
    }

    async function runPipeline(event) {
      event.preventDefault();

      const form = document.getElementById("pipelineForm");
      const formData = new FormData(form);

      const fileInput = document.getElementById("files");
      if (!fileInput.files || fileInput.files.length === 0) {
        setStatus("Please select at least one file.", "err");
        return;
      }

      setStatus("Running EvalForge pipeline. This may take a few seconds...", "info");

      try {
        const response = await fetch("/pipeline/run", {
          method: "POST",
          body: formData
        });

        const text = await response.text();

        let data;
        try {
          data = JSON.parse(text);
        } catch {
          throw new Error(text);
        }

        if (!response.ok) {
          throw new Error(JSON.stringify(data, null, 2));
        }

        setStatus("Pipeline completed successfully.", "ok");
        updateDashboard(data);
        await loadRuns();

      } catch (err) {
        setStatus("Pipeline failed: " + err.message, "err");
      }
    }

    async function loadCasesPreview(runId) {
      const el = document.getElementById("casesPreview");

      try {
        const response = await fetch(`/pipeline/runs/${runId}/cases`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Failed to load cases.");
        }

        const cases = data.cases || [];
        if (cases.length === 0) {
          el.innerHTML = "No cases found.";
          return;
        }

        el.innerHTML = cases.slice(0, 8).map((item) => `
          <div class="case-card">
            <div>
              <span class="pill">${item.test_type}</span>
              <span class="pill ${item.risk_level === "high" ? "red" : "green"}">${item.risk_level}</span>
              <span class="pill yellow">${item.review_status}</span>
            </div>
            <div class="case-query">${item.user_query || ""}</div>
            <div class="muted small"><strong>Expected:</strong> ${item.expected_behavior || ""}</div>
            <div class="muted small"><strong>Test ID:</strong> <code>${item.test_id}</code></div>
          </div>
        `).join("");

      } catch (err) {
        el.innerHTML = "Failed to load cases: " + err.message;
      }
    }

    async function loadRuns() {
      const el = document.getElementById("runsTable");

      try {
        const response = await fetch("/pipeline/runs?limit=20");
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Failed to load runs.");
        }

        const runs = data.runs || [];

        if (runs.length === 0) {
          el.innerHTML = "No persisted runs yet.";
          return;
        }

        el.innerHTML = `
          <table>
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Project</th>
                <th>Version</th>
                <th>Cases</th>
                <th>Validity</th>
                <th>Score</th>
                <th>Exports</th>
              </tr>
            </thead>
            <tbody>
              ${runs.map((run) => {
                const q = run.quality_summary || {};
                const e = run.eval_summary || {};
                return `
                  <tr>
                    <td><code>${run.run_id}</code></td>
                    <td>${run.project_id}</td>
                    <td>${run.dataset_version}</td>
                    <td>${run.case_count}</td>
                    <td>${formatNumber(q.validity_rate)}</td>
                    <td>${formatNumber(e.average_score)}</td>
                    <td>
                      <a href="/pipeline/runs/${run.run_id}/export/json" target="_blank">JSON</a>
                      |
                      <a href="/pipeline/runs/${run.run_id}/export/csv" target="_blank">CSV</a>
                      |
                      <a href="/pipeline/runs/${run.run_id}/cases" target="_blank">Cases</a>
                    </td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        `;

      } catch (err) {
        el.innerHTML = "Failed to load runs: " + err.message;
      }
    }

    async function compareRuns() {
      const currentRunId = document.getElementById("currentRunId").value.trim();
      const baselineRunId = document.getElementById("baselineRunId").value.trim();
      const el = document.getElementById("compareResult");

      if (!currentRunId || !baselineRunId) {
        el.innerHTML = "Enter both current and baseline run IDs.";
        return;
      }

      try {
        const response = await fetch(`/pipeline/runs/${currentRunId}/compare/${baselineRunId}`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Comparison failed.");
        }

        const avg = data.metric_comparison?.average_score || {};
        const pass = data.metric_comparison?.pass_rate || {};
        const validity = data.metric_comparison?.validity_rate || {};

        el.innerHTML = `
          <div class="case-card">
            <h3>${data.regression_detected ? "Regression Detected" : "No Regression Detected"}</h3>
            <p>
              <strong>Average Score Delta:</strong>
              <span class="${avg.delta >= 0 ? "compare-good" : "compare-bad"}">${avg.delta}</span>
            </p>
            <p>
              <strong>Pass Rate Delta:</strong>
              <span class="${pass.delta >= 0 ? "compare-good" : "compare-bad"}">${pass.delta}</span>
            </p>
            <p>
              <strong>Validity Rate Delta:</strong>
              <span class="${validity.delta >= 0 ? "compare-good" : "compare-bad"}">${validity.delta}</span>
            </p>
            <p><strong>Reasons:</strong></p>
            <pre>${JSON.stringify(data.regression_reasons || [], null, 2)}</pre>
          </div>
        `;

      } catch (err) {
        el.innerHTML = "Comparison failed: " + err.message;
      }
    }

    document.getElementById("pipelineForm").addEventListener("submit", runPipeline);
    window.addEventListener("load", loadRuns);
  </script>
</body>
</html>
    """