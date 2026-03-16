let dashboardMeta = window.__INITIAL_DASHBOARD__ || {};
let selectedModel = "strong_baseline";
let selectedDemoId = null;
let currentResult = null;
let comparisonResults = null;

const STATUS_STYLES = {
  occupied: { label: "Occupied", className: "occupied", note: "Likely taken by a vehicle" },
  free: { label: "Free", className: "free", note: "Likely available" },
  unknown: { label: "Unknown", className: "unknown", note: "Low-confidence or abstained prediction" },
};

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const node = byId(id);
  if (node) node.textContent = value;
}

function pct(value) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

function fmt(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function deriveSummary(data) {
  const total = Number(data.total_slots || 0);
  const occupied = Number(data.occupied || 0);
  const free = Number(data.free || 0);
  const unknown = Number(data.unknown || 0);
  const known = Number(data.known || occupied + free);
  const coverage = data.coverage != null ? Number(data.coverage) : total ? known / total : 0;
  const unknownRate = data.unknown_rate != null ? Number(data.unknown_rate) : total ? unknown / total : 0;
  const meanConfidence = data.mean_confidence != null
    ? Number(data.mean_confidence)
    : (data.slots || []).length
      ? (data.slots || []).reduce((acc, slot) => acc + Number(slot.confidence || 0), 0) / data.slots.length
      : 0;
  const occupancyKnown = data.occupancy_rate_known != null ? Number(data.occupancy_rate_known) : known ? occupied / known : 0;
  const lowConfidence = data.low_confidence_slots != null
    ? Number(data.low_confidence_slots)
    : (data.slots || []).filter((slot) => Number(slot.confidence || 0) < 0.6).length;
  const qualityBand = data.quality_band || (unknownRate > 0.3 ? "low" : meanConfidence < 0.72 ? "medium" : "high");
  return {
    ...data,
    known,
    coverage,
    unknown_rate: unknownRate,
    mean_confidence: meanConfidence,
    occupancy_rate_known: occupancyKnown,
    low_confidence_slots: lowConfidence,
    quality_band: qualityBand,
    quality_summary: data.quality_summary || `${data.model_label || data.model_key}: ${occupied} occupied, ${free} free, ${unknown} uncertain slots.`,
    occupancy_note: data.occupancy_note || (known ? `Among confident slots, occupancy is ${pct(occupancyKnown)}.` : "No confident slots in this result."),
    quality_note: data.quality_note || `Coverage ${pct(coverage)} over all slots, mean confidence ${pct(meanConfidence)}.`,
  };
}

function getModelCards() {
  return dashboardMeta.model_cards || [];
}

function getModelCard(modelKey) {
  return getModelCards().find((card) => card.model_key === modelKey);
}

function activeTab() {
  return document.querySelector(".tab-btn.active")?.dataset.tab || "demo";
}

function showError(message) {
  window.alert(message);
}

function setMeter(id, labelId, fraction) {
  const value = Math.max(0, Math.min(1, Number(fraction || 0)));
  const bar = byId(id);
  if (bar) bar.style.width = `${value * 100}%`;
  if (labelId) setText(labelId, pct(value));
}

function renderModelToggle() {
  const wrap = byId("modelToggle");
  if (!wrap) return;
  wrap.innerHTML = "";
  getModelCards().forEach((card) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `model-option ${card.model_key === selectedModel ? "active" : ""}`;
    button.dataset.model = card.model_key;
    button.innerHTML = `
      <div class="option-title">
        <strong>${card.label}</strong>
        <span class="pill">${card.benchmark_ready ? "benchmarked" : "no benchmark"}</span>
      </div>
      <span>${card.approach}</span>
      <span>${card.cost}</span>
    `;
    button.addEventListener("click", () => {
      selectedModel = card.model_key;
      renderModelToggle();
      renderSelectedModelCard();
      renderSelectedModelBlurb();
      if (comparisonResults?.[selectedModel]) {
        renderResult(comparisonResults[selectedModel], { preserveComparison: true });
      }
    });
    wrap.appendChild(button);
  });
}

function renderSelectedModelBlurb() {
  const card = getModelCard(selectedModel);
  if (!card) return;
  const benchmarkText = card.benchmark_ready
    ? `Benchmark F1 ${fmt(card.benchmark.f1_macro, 3)}, coverage ${pct(card.benchmark.coverage)}.`
    : "No held-out benchmark file available yet.";
  setText(
    "selectedModelBlurb",
    `${card.label}: ${card.strength}. Trade-off: ${card.cost}. ${benchmarkText}`,
  );
}

function renderModelCatalog() {
  const wrap = byId("modelCards");
  if (!wrap) return;
  wrap.innerHTML = "";
  const cards = getModelCards();
  if (!cards.length) {
    wrap.innerHTML = `<div class="model-card"><p>No model metadata available.</p></div>`;
    return;
  }
  cards.forEach((card) => {
    const benchmark = card.benchmark_ready
      ? `Held-out F1 ${fmt(card.benchmark.f1_macro, 3)} · coverage ${pct(card.benchmark.coverage)}`
      : "Held-out benchmark not available";
    const artifactClass = card.artifact_ready ? "status-ready" : "status-missing";
    const runSummary = card.run_summary?.best_f1_macro
      ? `Last training best F1: ${fmt(card.run_summary.best_f1_macro, 3)}`
      : card.run_summary?.best_val_f1_known
        ? `Last validation covered F1: ${fmt(card.run_summary.best_val_f1_known, 3)}`
        : card.run_summary?.best_val_accuracy
          ? `Last validation accuracy: ${fmt(card.run_summary.best_val_accuracy, 3)}`
          : "No local run summary found";
    const artifactMode = card.artifact_meta?.has_occupied_threshold || card.artifact_meta?.has_unknown_threshold
      ? "Calibrated artifact"
      : card.model_key === "fast_classic"
        ? "Legacy artifact"
        : "Default artifact";
    const item = document.createElement("article");
    item.className = "model-card";
    item.innerHTML = `
      <h3>${card.label}</h3>
      <p><strong>Approach:</strong> ${card.approach}</p>
      <p><strong>Best use:</strong> ${card.strength}</p>
      <p><strong>Trade-off:</strong> ${card.cost}</p>
      <p class="${artifactClass}">${card.artifact_ready ? "Artifact ready" : "Artifact missing"}</p>
      <p>${artifactMode}</p>
      <p>${benchmark}</p>
      <p>${runSummary}</p>
    `;
    wrap.appendChild(item);
  });
}

function renderSelectedModelCard() {
  const wrap = byId("modelInfoCards");
  if (!wrap) return;
  wrap.innerHTML = "";
  const card = getModelCard(selectedModel);
  if (!card) {
    wrap.innerHTML = `<div class="info-card"><p>No selected model metadata.</p></div>`;
    return;
  }
  const benchmark = card.benchmark;
  const qualityBadge = benchmark
    ? benchmark.f1_macro >= 0.9
      ? ["good", "Strong test quality"]
      : benchmark.f1_macro >= 0.8
        ? ["warn", "Moderate test quality"]
        : ["bad", "Weak test quality"]
    : ["warn", "No test benchmark"];
  const runSummary = card.run_summary || {};
  const artifactMeta = card.artifact_meta || {};
  const artifactMode = artifactMeta.has_occupied_threshold || artifactMeta.has_unknown_threshold
    ? "calibrated"
    : card.model_key === "fast_classic"
      ? "legacy"
      : "default";
  const item = document.createElement("article");
  item.className = "info-card";
  item.innerHTML = `
    <header>
      <strong>${card.label}</strong>
      <span class="badge ${qualityBadge[0]}">${qualityBadge[1]}</span>
    </header>
    <div class="info-card-grid">
      <p><strong>Approach:</strong> ${card.approach}</p>
      <p><strong>Strength:</strong> ${card.strength}</p>
      <p><strong>Trade-off:</strong> ${card.cost}</p>
      <p><strong>Artifact:</strong> ${card.artifact_ready ? "ready" : "missing"}</p>
      <p><strong>Artifact mode:</strong> ${artifactMode}</p>
      <p><strong>Held-out benchmark:</strong> ${
        benchmark
          ? `accuracy ${fmt(benchmark.accuracy, 3)}, F1 ${fmt(benchmark.f1_macro, 3)}, coverage ${pct(benchmark.coverage)}`
          : "not available"
      }</p>
      <p><strong>Local run summary:</strong> ${
        runSummary.best_f1_macro
          ? `best training F1 ${fmt(runSummary.best_f1_macro, 3)}`
          : runSummary.best_val_f1_known
            ? `covered validation F1 ${fmt(runSummary.best_val_f1_known, 3)}`
          : runSummary.best_val_accuracy
            ? `best validation accuracy ${fmt(runSummary.best_val_accuracy, 3)}`
            : "no summary artifact"
      }</p>
    </div>
  `;
  wrap.appendChild(item);
}

function renderDemoFeed() {
  const wrap = byId("demoFeed");
  if (!wrap) return;
  wrap.innerHTML = "";
  const items = dashboardMeta.demo_feed || [];
  if (!items.length) {
    wrap.innerHTML = `<div class="model-card"><p>No demo media available.</p></div>`;
    return;
  }
  items.forEach((item, index) => {
    const card = document.createElement("article");
    const active = item.id === selectedDemoId || (!selectedDemoId && index === 0);
    if (active) selectedDemoId = item.id;
    card.className = `demo-card ${active ? "active" : ""}`;
    card.dataset.demoId = item.id;
    card.innerHTML = `
      <div class="demo-thumb">
        <img src="${item.poster_url || item.media_url}" alt="${item.title}" />
      </div>
      <div class="demo-body">
        <div class="demo-meta">
          <span class="pill">${item.kind}</span>
          <span class="pill">${item.scene}</span>
          <span class="pill">${item.weather}</span>
        </div>
        <div>
          <h3 class="demo-title">${item.title}</h3>
          <p class="demo-subtitle">${item.subtitle || ""}</p>
        </div>
      </div>
    `;
    card.addEventListener("click", () => {
      selectedDemoId = item.id;
      byId("selectedDemoId").value = item.id;
      document.querySelectorAll(".demo-card").forEach((node) => node.classList.remove("active"));
      card.classList.add("active");
    });
    wrap.appendChild(card);
  });
  byId("selectedDemoId").value = selectedDemoId || "";
}

function renderBenchmarkTable() {
  const evaluation = dashboardMeta.evaluation || {};
  setText("topEvalScope", evaluation.scope || "No evaluation loaded");
  setText("evalBadge", evaluation.ready ? "Held-out metrics loaded" : "No evaluation loaded");
  setText(
    "benchmarkCopy",
    evaluation.ready
      ? "These numbers come from the last held-out evaluation run in runs/eval/latest. Unknown predictions reduce coverage and are shown separately."
      : "No runs/eval/latest summary.csv found. The app will show model artifacts and local run summaries, but not claim held-out quality without a proper benchmark.",
  );

  const wrap = byId("comparisonTableWrap");
  const charts = byId("comparisonCharts");
  if (!wrap || !charts) return;
  wrap.innerHTML = "";
  charts.innerHTML = "";

  if (!evaluation.ready || !evaluation.models?.length) {
    wrap.innerHTML = `<div class="side-card"><p>Run the evaluation pipeline to populate held-out benchmark metrics for both models.</p></div>`;
    return;
  }

  const table = document.createElement("table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Model</th>
        <th>Accuracy</th>
        <th>Accuracy on covered</th>
        <th>Coverage</th>
        <th>Unknown rate</th>
        <th>Macro F1</th>
        <th>Latency</th>
        <th>FPS</th>
        <th>RSS</th>
        <th>Samples</th>
      </tr>
    </thead>
    <tbody>
      ${evaluation.models
        .map(
          (row) => `
            <tr>
              <td><strong>${row.label}</strong></td>
              <td>${fmt(row.accuracy, 3)}</td>
              <td>${fmt(row.accuracy_known, 3)}</td>
              <td>${pct(row.coverage)}</td>
              <td>${pct(row.unknown_rate)}</td>
              <td>${fmt(row.f1_macro, 3)}</td>
              <td>${fmt(row.latency_ms_mean, 2)} ms</td>
              <td>${fmt(row.fps_estimate, 2)}</td>
              <td>${fmt(row.rss_mb_mean, 1)} MB</td>
              <td>${row.num_samples}</td>
            </tr>
          `,
        )
        .join("")}
    </tbody>
  `;
  wrap.appendChild(table);

  Object.entries(evaluation.charts || {}).forEach(([key, url]) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `<img src="${url}" alt="${key}" />`;
    charts.appendChild(card);
  });
}

function slotSignal(slot) {
  if (slot.status === "unknown") return "abstained";
  if ((slot.confidence || 0) < 0.6) return "weak";
  if ((slot.confidence || 0) < 0.8) return "moderate";
  return "strong";
}

function renderSlotTable(slots = []) {
  const body = byId("slotTableBody");
  if (!body) return;
  body.innerHTML = "";
  if (!slots.length) {
    body.innerHTML = `<tr><td colspan="4">No slot-level output yet.</td></tr>`;
    return;
  }
  slots.forEach((slot) => {
    const meta = STATUS_STYLES[slot.status] || STATUS_STYLES.unknown;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${slot.slot_id}</td>
      <td>
        <span class="status-chip">
          <span class="status-dot ${meta.className}"></span>
          ${meta.label}
        </span>
      </td>
      <td>${fmt(slot.confidence, 3)}</td>
      <td>${slotSignal(slot)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderPreview(data) {
  const wrap = byId("previewWrap");
  if (!wrap) return;
  wrap.innerHTML = "";
  if (!data?.output_path) {
    wrap.innerHTML = `
      <div class="preview-empty">
        <strong>No preview generated</strong>
        <span>The run finished without a renderable output file.</span>
      </div>
    `;
    return;
  }
  if (data.output_path.endsWith(".mp4")) {
    const video = document.createElement("video");
    video.src = data.output_path;
    video.controls = true;
    wrap.appendChild(video);
    return;
  }
  const img = document.createElement("img");
  img.src = data.output_path;
  img.alt = `${data.model_label || data.model_key} inference output`;
  wrap.appendChild(img);
}

function renderLiveSummary(data) {
  setText("runHeadline", `${data.model_label || data.model_key} on ${data.demo_title || data.source_kind || "input media"}`);
  setText(
    "runSummary",
    `${data.quality_summary} ${data.occupancy_note} ${data.quality_note}`,
  );
  setText("currentTotal", data.total_slots);
  setText("currentOccupied", data.occupied);
  setText("currentFree", data.free);
  setText("currentUnknown", data.unknown);
  setText("currentCoverage", pct(data.coverage));
  setText("currentOccKnown", pct(data.occupancy_rate_known));
  setText("currentLatency", `${fmt(data.latency_ms, 2)} ms`);
  setText("currentFPS", fmt(data.fps_estimate, 2));
  setText("currentConfidence", pct(data.mean_confidence));
  setText("currentLowConfidence", data.low_confidence_slots);
  setText("currentQualityBand", data.quality_band);
  setText("currentUnknownRate", pct(data.unknown_rate));
  setText("qualitySummary", data.quality_summary);
  setText("qualityNote", `${data.occupancy_note} ${data.quality_note}`);
  setMeter("barCoverage", "coverageLabel", data.coverage);
  setMeter("barConfidence", "confidenceLabel", data.mean_confidence);
  setMeter("barUncertainty", "uncertaintyLabel", data.unknown_rate);
}

function renderComparisonStrip() {
  const wrap = byId("comparisonStrip");
  if (!wrap) return;
  if (!comparisonResults) {
    wrap.classList.add("hidden");
    wrap.innerHTML = "";
    return;
  }

  const keys = Object.keys(comparisonResults);
  if (keys.length < 2) {
    wrap.classList.add("hidden");
    wrap.innerHTML = "";
    return;
  }

  const [left, right] = keys.map((key) => comparisonResults[key]);
  const disagreements = [];
  const rightMap = new Map((right.slots || []).map((slot) => [slot.slot_id, slot.status]));
  (left.slots || []).forEach((slot) => {
    if (rightMap.get(slot.slot_id) && rightMap.get(slot.slot_id) !== slot.status) {
      disagreements.push(slot.slot_id);
    }
  });

  wrap.classList.remove("hidden");
  wrap.innerHTML = `
    <div class="comparison-results">
      ${[left, right]
        .map(
          (item) => `
            <article class="comparison-result-card ${item.model_key === selectedModel ? "active" : ""}">
              <header>
                <strong>${item.model_label || item.model_key}</strong>
                <button class="ghost-btn" type="button" data-switch-model="${item.model_key}">Show overlay</button>
              </header>
              <div class="comparison-result-grid">
                <div class="comparison-kpi">Occupied <strong>${item.occupied}</strong></div>
                <div class="comparison-kpi">Free <strong>${item.free}</strong></div>
                <div class="comparison-kpi">Unknown <strong>${item.unknown}</strong></div>
                <div class="comparison-kpi">Coverage <strong>${pct(item.coverage)}</strong></div>
                <div class="comparison-kpi">Latency <strong>${fmt(item.latency_ms, 1)} ms</strong></div>
                <div class="comparison-kpi">Mean confidence <strong>${pct(item.mean_confidence)}</strong></div>
              </div>
            </article>
          `,
        )
        .join("")}
    </div>
    <div class="comparison-disagreement">
      <strong>Disagreement rate:</strong> ${left.total_slots ? pct(disagreements.length / left.total_slots) : "0.0%"}
      <br />
      <span>${
        disagreements.length
          ? `Slots with different labels: ${disagreements.slice(0, 20).join(", ")}${disagreements.length > 20 ? "..." : ""}`
          : "Both models agree on every visible slot in this run."
      }</span>
    </div>
  `;

  wrap.querySelectorAll("[data-switch-model]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedModel = button.dataset.switchModel;
      renderModelToggle();
      renderSelectedModelCard();
      renderSelectedModelBlurb();
      if (comparisonResults[selectedModel]) {
        renderResult(comparisonResults[selectedModel], { preserveComparison: true });
      }
    });
  });
}

function renderResult(data, options = {}) {
  currentResult = deriveSummary(data);
  renderPreview(currentResult);
  renderLiveSummary(currentResult);
  renderSlotTable(currentResult.slots || []);
  if (!options.preserveComparison) {
    comparisonResults = null;
  }
  renderComparisonStrip();
}

function getCurrentLayoutPath(fallback = "") {
  return byId("layoutPath")?.value || fallback;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function runDemoPrediction() {
  const demoId = selectedDemoId || byId("selectedDemoId").value;
  if (!demoId) {
    showError("Select a demo first.");
    return;
  }
  const compare = byId("compareDemo")?.checked;
  const makeBody = (modelKey) => JSON.stringify({
    demo_id: demoId,
    model_key: modelKey,
    max_frames: 120,
    stride: 4,
  });

  if (compare) {
    const cards = getModelCards();
    const results = await Promise.all(
      cards.map((card) =>
        requestJson("/predict_demo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: makeBody(card.model_key),
        }),
      ),
    );
    comparisonResults = Object.fromEntries(results.map((item) => [item.model_key, item]));
    renderComparisonStrip();
    renderResult(comparisonResults[selectedModel] || results[0], { preserveComparison: true });
    return;
  }

  const payload = await requestJson("/predict_demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: makeBody(selectedModel),
  });
  renderResult(payload);
}

async function predictImage() {
  const file = byId("imageFile")?.files?.[0];
  if (!file) {
    showError("Choose an image file first.");
    return;
  }
  const form = new FormData();
  form.append("image", file);
  form.append("model_key", selectedModel);
  form.append("layout_path", getCurrentLayoutPath());
  const payload = await requestJson("/predict_image", { method: "POST", body: form });
  renderResult(payload);
}

async function predictVideo() {
  const file = byId("videoFile")?.files?.[0];
  if (!file) {
    showError("Choose a video file first.");
    return;
  }
  const form = new FormData();
  form.append("video", file);
  form.append("model_key", selectedModel);
  form.append("layout_path", getCurrentLayoutPath());
  form.append("max_frames", byId("videoMaxFrames").value || "180");
  form.append("stride", byId("videoStride").value || "5");
  const payload = await requestJson("/predict_video", { method: "POST", body: form });
  renderResult(payload);
}

async function predictStream() {
  const rtspUrl = byId("streamUrl")?.value;
  if (!rtspUrl) {
    showError("Enter an RTSP or stream URL.");
    return;
  }
  const payload = await requestJson("/predict_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      rtsp_url: rtspUrl,
      layout_path: getCurrentLayoutPath(),
      model_key: selectedModel,
      max_frames: Number(byId("streamMaxFrames").value || 120),
      stride: Number(byId("streamStride").value || 5),
    }),
  });
  renderResult(payload);
}

function exportCurrent(kind) {
  if (!currentResult) {
    showError("Run a prediction first.");
    return;
  }
  if (kind === "json") {
    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: "application/json" });
    return downloadBlob(blob, "parking_vision_result.json");
  }
  const header = "slot_id,status,confidence\n";
  const rows = (currentResult.slots || [])
    .map((slot) => `${slot.slot_id},${slot.status},${slot.confidence}`)
    .join("\n");
  downloadBlob(new Blob([header + rows], { type: "text/csv" }), "parking_vision_slots.csv");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((node) => node.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      document.querySelector(`[data-panel="${button.dataset.tab}"]`)?.classList.add("active");
    });
  });
}

function setupTheme() {
  byId("themeToggle")?.addEventListener("click", () => {
    document.body.dataset.theme = document.body.dataset.theme === "dark" ? "light" : "dark";
  });
}

function setupActions() {
  byId("runDemoBtn")?.addEventListener("click", () => runDemoPrediction().catch((error) => showError(error.message)));
  byId("runImageBtn")?.addEventListener("click", () => predictImage().catch((error) => showError(error.message)));
  byId("runVideoBtn")?.addEventListener("click", () => predictVideo().catch((error) => showError(error.message)));
  byId("runStreamBtn")?.addEventListener("click", () => predictStream().catch((error) => showError(error.message)));
  byId("exportJsonBtn")?.addEventListener("click", () => exportCurrent("json"));
  byId("exportCsvBtn")?.addEventListener("click", () => exportCurrent("csv"));
}

async function loadDashboardMeta() {
  try {
    dashboardMeta = await requestJson("/dashboard_meta", {});
  } catch (error) {
    console.warn("Failed to refresh dashboard metadata", error);
  }
  renderModelToggle();
  renderSelectedModelBlurb();
  renderModelCatalog();
  renderSelectedModelCard();
  renderDemoFeed();
  renderBenchmarkTable();
}

function init() {
  setupTabs();
  setupTheme();
  setupActions();
  renderModelToggle();
  renderSelectedModelBlurb();
  renderModelCatalog();
  renderSelectedModelCard();
  renderDemoFeed();
  renderBenchmarkTable();
  loadDashboardMeta();
}

window.addEventListener("DOMContentLoaded", init);
