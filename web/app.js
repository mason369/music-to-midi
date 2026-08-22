const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const LOCALE_FILES = {
  zh_CN: "locales/zh_CN.json",
  en_US: "locales/en_US.json",
};

const FRONTEND_API_VERSION = "2.0";
const BACKEND_HEARTBEAT_INTERVAL_MS = 5000;

const TRACK_COLORS = ["#c89b55", "#50b7a3", "#be7058", "#8298b7", "#9b7aa5", "#8faf69", "#c77988", "#6ca2ad"];
const STARTUP_API_QUERY = new URLSearchParams(window.location.search).get("api");
const STARTUP_STORED_API = localStorage.getItem("musicToMidiApiBase");

function normalizeApiBase(value) {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error(`Unsupported API URL protocol: ${url.protocol}`);
  return url.href.replace(/\/$/, "");
}

function initialApiBase() {
  if (STARTUP_API_QUERY) {
    const normalized = normalizeApiBase(STARTUP_API_QUERY);
    localStorage.setItem("musicToMidiApiBase", normalized);
    return normalized;
  }
  if (STARTUP_STORED_API) return normalizeApiBase(STARTUP_STORED_API);
  if (["http:", "https:"].includes(window.location.protocol)) {
    const port = window.location.port === "5173" ? "8765" : window.location.port;
    return `${window.location.protocol}//${window.location.hostname}${port ? `:${port}` : ""}`;
  }
  return "http://127.0.0.1:8765";
}

const state = {
  language: localStorage.getItem("musicToMidiLanguage") || "zh_CN",
  messages: null,
  apiBase: initialApiBase(),
  frontendUrl: window.location.origin,
  capabilities: null,
  selectedMode: "smart",
  audioFile: null,
  sourceObjectUrl: "",
  currentJob: null,
  submissionPending: false,
  connectionStatus: "connecting",
  connectionError: "",
  expectedApiVersion: FRONTEND_API_VERSION,
  heartbeatPending: false,
  progressStageSignature: "",
  guideAction: "source",
  eventSources: new Map(),
  tracks: [],
  audioContext: null,
  activeSources: [],
  playing: false,
  playhead: 0,
  playStartedAt: 0,
  raf: 0,
  zoom: 1,
};

async function loadFrontendRuntimeConfig() {
  const response = await fetch("runtime-config.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`runtime-config.json: HTTP ${response.status}`);
  const config = await response.json();
  if (typeof config !== "object" || config === null) throw new Error("runtime-config.json must contain an object");
  if (config.expected_api_version !== FRONTEND_API_VERSION) {
    throw new Error(`Frontend runtime contract mismatch: expected ${FRONTEND_API_VERSION}, configured ${config.expected_api_version || "missing"}`);
  }
  state.expectedApiVersion = config.expected_api_version;
  state.frontendUrl = normalizeApiBase(config.frontend_url);
  const configuredBackend = normalizeApiBase(config.backend_url);
  if (!STARTUP_API_QUERY && !STARTUP_STORED_API) state.apiBase = configuredBackend;
}

const TERMINAL_JOB_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

async function loadLocaleCatalogs() {
  const entries = await Promise.all(Object.entries(LOCALE_FILES).map(async ([locale, path]) => {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return [locale, await response.json()];
  }));
  const catalogs = Object.fromEntries(entries);
  const expected = Object.keys(catalogs.zh_CN).sort();
  for (const [locale, catalog] of Object.entries(catalogs)) {
    const keys = Object.keys(catalog).sort();
    if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
      throw new Error(`Locale catalog key mismatch: ${locale}`);
    }
    if (Object.values(catalog).some((value) => typeof value !== "string")) {
      throw new Error(`Locale catalog contains non-string values: ${locale}`);
    }
  }
  state.messages = catalogs;
  if (!state.messages[state.language]) state.language = "zh_CN";
}

function t(key, variables = {}) {
  const catalog = state.messages?.[state.language];
  if (!catalog || !Object.prototype.hasOwnProperty.call(catalog, key)) {
    throw new Error(`Missing translation: ${state.language}.${key}`);
  }
  return catalog[key].replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(variables, name) ? String(variables[name]) : match
  ));
}

function isJobRunning(job = state.currentJob) {
  return Boolean(job && !TERMINAL_JOB_STATUSES.has(job.status));
}

function progressStageDefinitions(mode = state.selectedMode) {
  if (["vocal_split", "six_stem_split"].includes(mode)) {
    return [
      ["queued", "progress.stage.queued"],
      ["preprocessing", "progress.stage.preprocessing"],
      ["separation", "progress.stage.separation"],
      ["complete", "progress.stage.complete"],
    ];
  }
  return [
    ["queued", "progress.stage.queued"],
    ["preprocessing", "progress.stage.preprocessing"],
    ["transcription", "progress.stage.transcription"],
    ["synthesis", "progress.stage.synthesis"],
    ["complete", "progress.stage.complete"],
  ];
}

function renderProgressStages(mode = state.selectedMode) {
  const definitions = progressStageDefinitions(mode);
  const signature = `${state.language}:${mode}:${definitions.map(([stage, key]) => `${stage}:${key}`).join("|")}`;
  if (signature === state.progressStageSignature) return definitions;
  state.progressStageSignature = signature;
  $("#stageRow").innerHTML = definitions.map(([stage, key]) => (
    `<span data-stage="${stage}">${escapeHtml(t(key))}</span>`
  )).join("");
  $("#stageRow").style.setProperty("--stage-count", String(definitions.length));
  return definitions;
}

function workflowMode(job = state.currentJob) {
  return job?.request?.processing_mode || job?.result?.mode || state.selectedMode;
}

function currentProgressLabel(job = state.currentJob) {
  const mode = workflowMode(job);
  const stage = job?.status === "queued" ? "queued" : (job?.progress?.stage || job?.status || "queued");
  const definitions = progressStageDefinitions(mode);
  const match = definitions.find(([candidate]) => candidate === stage);
  if (match) return t(match[1]);
  const key = `progress.stage.${stage}`;
  return state.messages[state.language][key] ? t(key) : t(`job.status.${job?.status || "queued"}`);
}

function focusWorkflowTargets(focus) {
  const targetIds = {
    source: ["dropZone"],
    configure: ["configurationPanel", "actionDeck"],
    process: ["progressPanel"],
    result: ["resultPanel"],
  };
  document.body.dataset.workflowFocus = focus;
  $$(".guidance-target").forEach((node) => {
    node.classList.toggle("is-guided", (targetIds[focus] || []).includes(node.id) && !node.hidden);
  });
}

function updateWorkflowGuide() {
  if (!state.messages) return;
  const job = state.currentJob;
  const mode = workflowMode(job);
  const modeName = modeLabel(mode);
  let focus = "source";
  let title = "";
  let description = "";
  let action = "source";
  let actionLabel = t("guide.action.source");

  if (!state.capabilities) {
    focus = "connect";
    title = t("guide.connect.title");
    description = t("guide.connect.description");
    action = "connect";
    actionLabel = t("guide.action.connect");
  } else if (state.submissionPending) {
    focus = "process";
    title = t("guide.submitting.title");
    description = t("guide.submitting.description");
    action = "progress";
    actionLabel = t("guide.action.progress");
  } else if (isJobRunning(job)) {
    focus = "process";
    title = t("guide.running.title", { stage: currentProgressLabel(job) });
    description = t("guide.running.description");
    action = "progress";
    actionLabel = t("guide.action.progress");
  } else if (job?.status === "succeeded") {
    focus = "result";
    action = "result";
    actionLabel = t("guide.action.result");
    if (job.result?.manual_midi_required) {
      title = t("guide.result.separation_title");
      description = t("guide.result.separation_description");
    } else {
      title = t("guide.result.midi_title");
      description = t("guide.result.midi_description");
    }
  } else if (job?.status === "failed") {
    focus = "result";
    title = t("guide.result.failed_title");
    description = t("guide.result.failed_description");
    action = "result";
    actionLabel = t("guide.action.result");
  } else if (job?.status === "cancelled") {
    focus = "result";
    title = t("guide.result.cancelled_title");
    description = t("guide.result.cancelled_description");
    action = "result";
    actionLabel = t("guide.action.result");
  } else if (!state.audioFile) {
    title = t("guide.source.title");
    description = t("guide.source.description", { mode: modeName });
  } else {
    focus = "configure";
    action = "configure";
    actionLabel = t("guide.action.configure");
    title = t("guide.configure.title");
    description = t("guide.configure.description", { file: state.audioFile.name, mode: modeName });
  }

  const activeStep = focus === "connect" ? "route" : focus;
  const steps = ["route", "source", "configure", "process", "result"];
  const activeIndex = steps.indexOf(activeStep);
  $$('[data-workflow-step]').forEach((item) => {
    const index = steps.indexOf(item.dataset.workflowStep);
    const button = $("button", item);
    item.classList.toggle("is-active", index === activeIndex);
    item.classList.toggle("is-complete", index >= 0 && index < activeIndex);
    button.setAttribute("aria-current", index === activeIndex ? "step" : "false");
  });
  $("#guideTitle").textContent = title;
  $("#guideDescription").textContent = description;
  $("#guideActionLabel").textContent = actionLabel;
  $("#guideAction").disabled = state.submissionPending;
  state.guideAction = action;
  focusWorkflowTargets(focus);
}

function scrollToWorkflowTarget(target) {
  const selectors = {
    route: "#modeSelect",
    source: "#dropZone",
    configure: "#configurationPanel",
    process: "#progressPanel:not([hidden]), #actionDeck",
    result: "#resultPanel:not([hidden]), #actionDeck",
  };
  const node = $(selectors[target] || selectors.source);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "center" });
  if (target === "configure") {
    const preferred = $("select:not([hidden])", node);
    preferred?.focus({ preventScroll: true });
  } else if (node.matches("button, input, select, [tabindex]")) {
    node.focus({ preventScroll: true });
  }
}

function performGuideAction() {
  if (state.guideAction === "connect") {
    openApiDialog();
  } else if (state.guideAction === "source") {
    $("#audioInput").click();
  } else {
    scrollToWorkflowTarget(state.guideAction);
  }
}

function prepareNewRequest() {
  if (!state.currentJob || isJobRunning()) return;
  state.currentJob = null;
  state.progressStageSignature = "";
  $("#progressPanel").hidden = true;
  resetResult();
}

function renderConnectionState() {
  const chip = $("#connectionButton");
  const online = state.connectionStatus === "online" && state.capabilities;
  const failed = state.connectionStatus === "unavailable";
  chip.className = `connection-chip ${online ? "is-online" : failed ? "is-error" : "is-pending"}`;
  $("#connectionLabel").textContent = t(`connection.${online ? "online" : failed ? "unavailable" : "connecting"}`);
  $("#backendDot").classList.toggle("online", Boolean(online));

  if (online) {
    const runtime = state.capabilities.runtime;
    const accelerator = String(runtime.accelerator || "").toLowerCase();
    const acceleratorLabel = accelerator === "xpu"
      ? "Intel XPU"
      : accelerator === "cuda" ? "CUDA" : accelerator.toUpperCase();
    const deviceRecords = Array.isArray(runtime.accelerator_devices) ? runtime.accelerator_devices : [];
    const firstRecord = deviceRecords[0];
    const deviceName = typeof firstRecord === "string" ? firstRecord : firstRecord?.name;
    const deviceIdentity = deviceName || runtime.accelerator_device || acceleratorLabel;
    const ready = runtime.accelerator_ready === true;
    $("#deviceName").textContent = ready && deviceIdentity ? deviceIdentity : t("backend.accelerator_unavailable");
    $("#runtimeVersion").textContent = `Torch ${runtime.torch} · API ${state.capabilities.api_version}`;
    $("#computeDevice").textContent = ready && deviceIdentity
      ? `${acceleratorLabel} / ${deviceIdentity}`
      : t("backend.accelerator_unavailable");
    return;
  }

  $("#deviceName").textContent = failed ? t("backend.connect_failed") : t("backend.waiting");
  $("#runtimeVersion").textContent = failed && state.connectionError ? state.connectionError : "—";
  $("#computeDevice").textContent = failed ? t("backend.accelerator_unavailable") : t("backend.waiting");
}

function applyLanguage({ rerender = true } = {}) {
  document.documentElement.lang = state.language === "zh_CN" ? "zh-CN" : "en";
  document.title = t("document.title");
  $$('[data-i18n]').forEach((node) => { node.textContent = t(node.dataset.i18n); });
  $$('[data-i18n-title]').forEach((node) => { node.title = t(node.dataset.i18nTitle); });
  $$('[data-i18n-aria-label]').forEach((node) => { node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel)); });
  $$('[data-i18n-placeholder]').forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  $("#languageButton").textContent = t("language.next");
  renderConnectionState();
  state.progressStageSignature = "";
  renderProgressStages(workflowMode());
  updateWorkflowGuide();
  if (!rerender) return;
  if (state.capabilities) populateControls();
  renderModes();
  updateConditionalControls();
  updateStartLabel();
  updateReadyState();
  if (state.currentJob) {
    updateProgress(state.currentJob);
    if (state.currentJob.status === "succeeded") renderPrimaryResult(state.currentJob, { restoreTracks: false, scroll: false });
    else if (state.currentJob.status === "failed") renderFailure(state.currentJob.error || t("error.backend_inference"), true, false);
    else if (state.currentJob.status === "cancelled") renderFailure(t("error.task_cancelled"), false, false);
  }
  if (state.tracks.length) renderMixer({ scroll: false });
  refreshJobs();
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let size = value; let index = 0;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size.toFixed(index ? 1 : 0)} ${units[index]}`;
}
function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const ms = Math.floor((safe % 1) * 1000);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}
function toast(message, kind = "info") {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  $("#toastStack").append(node);
  setTimeout(() => node.remove(), 6500);
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${state.apiBase}${path}`, options);
  } catch (error) {
    markBackendUnavailable(error);
    throw error;
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || payload);
    } catch { /* the HTTP status remains the explicit error */ }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}
function artifactUrl(artifact) { return `${state.apiBase}${artifact.download_url}`; }

function assertCompatibleApiVersion(payload, source) {
  const actual = String(payload?.api_version || "");
  if (actual !== state.expectedApiVersion) {
    throw new Error(t("error.api_version_mismatch", {
      source,
      expected: state.expectedApiVersion,
      actual: actual || t("error.api_version_missing"),
    }));
  }
}

function markBackendUnavailable(error) {
  state.capabilities = null;
  state.connectionStatus = "unavailable";
  state.connectionError = error instanceof Error ? error.message : String(error);
  if (state.messages) {
    renderConnectionState();
    updateReadyState();
  }
}

function endpointParts(value) {
  const url = new URL(normalizeApiBase(value));
  return {
    protocol: url.protocol,
    host: url.hostname,
    port: url.port || (url.protocol === "https:" ? "443" : "80"),
  };
}

function apiBaseFromDialog() {
  const protocol = $("#apiProtocol").value;
  const hostValue = $("#apiHost").value.trim();
  const port = Number($("#apiPort").value);
  if (!hostValue) throw new Error(t("dialog.host_required"));
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(t("dialog.port_invalid"));
  const host = hostValue.includes(":") && !hostValue.startsWith("[") ? `[${hostValue}]` : hostValue;
  return normalizeApiBase(`${protocol}//${host}:${port}`);
}

function updateEffectiveApiUrl() {
  const output = $("#effectiveApiUrl");
  try {
    output.textContent = apiBaseFromDialog();
  } catch (error) {
    output.textContent = error.message;
  }
}

function openApiDialog() {
  const parts = endpointParts(state.apiBase);
  $("#frontendAddress").value = state.frontendUrl;
  $("#apiProtocol").value = parts.protocol;
  $("#apiHost").value = parts.host;
  $("#apiPort").value = parts.port;
  const result = $("#connectionTestResult");
  result.className = "connection-test-result";
  result.textContent = t("dialog.test_idle");
  updateEffectiveApiUrl();
  $("#apiDialog").showModal();
}

async function testDialogConnection() {
  const result = $("#connectionTestResult");
  const button = $("#testApiConnection");
  button.disabled = true;
  result.className = "connection-test-result";
  result.textContent = t("dialog.testing");
  try {
    const base = apiBaseFromDialog();
    const response = await fetch(`${base}/api/v1/health`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
    const payload = await response.json();
    if (payload.status !== "ok") throw new Error(t("dialog.health_invalid"));
    assertCompatibleApiVersion(payload, "health");
    result.className = "connection-test-result is-success";
    result.textContent = t("dialog.test_success", { version: payload.api_version });
  } catch (error) {
    result.className = "connection-test-result is-error";
    result.textContent = t("dialog.test_failed", { error: error.message });
  } finally {
    button.disabled = false;
  }
}

async function connectBackend({ quiet = false } = {}) {
  state.connectionStatus = "connecting";
  state.connectionError = "";
  renderConnectionState();
  try {
    const [health, capabilities] = await Promise.all([api("/api/v1/health"), api("/api/v1/capabilities")]);
    if (health.status !== "ok") throw new Error(t("dialog.health_invalid"));
    assertCompatibleApiVersion(health, "health");
    assertCompatibleApiVersion(capabilities, "capabilities");
    state.capabilities = capabilities;
    state.connectionStatus = "online";
    renderConnectionState();
    populateControls();
    renderModes();
    const jobs = await refreshJobs();
    const activeJob = jobs.find((job) => !["succeeded", "failed", "cancelled"].includes(job.status));
    if (activeJob && !state.currentJob) openJob(activeJob, { scroll: false });
    updateReadyState();
  } catch (error) {
    markBackendUnavailable(error);
    if (!quiet) toast(t("error.backend_connection", { error: error.message }), "error");
  }
}

async function probeBackend() {
  if (state.heartbeatPending) return;
  state.heartbeatPending = true;
  try {
    const response = await fetch(`${state.apiBase}/api/v1/health`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
    const health = await response.json();
    if (health.status !== "ok") throw new Error(t("dialog.health_invalid"));
    assertCompatibleApiVersion(health, "health");
    if (state.connectionStatus !== "online" || !state.capabilities) {
      await connectBackend({ quiet: true });
    }
  } catch (error) {
    markBackendUnavailable(error);
  } finally {
    state.heartbeatPending = false;
  }
}

function option(select, value, label = value, availability = {}) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = availability.available === false ? `${label} · ${t("model.unavailable")}` : label;
  node.disabled = availability.available === false;
  if (availability.unavailable_reason) node.title = availability.unavailable_reason;
  select.append(node);
}
function selectAvailableOption(select, preferredValues = []) {
  const options = [...select.options];
  const preferred = preferredValues.find((value) => options.some((item) => item.value === value && !item.disabled));
  const target = preferred ? options.find((item) => item.value === preferred) : options.find((item) => !item.disabled);
  if (target) select.value = target.value;
  else select.selectedIndex = -1;
}
function populateControls() {
  const caps = state.capabilities;
  if (!caps) return;
  const backendLabels = { yourmt3: "YourMT3+", miros: "MIROS MusicFM", muscriptor: "MuScriptor" };
  const backend = $("#backendSelect"); const selectedBackend = backend.value; backend.innerHTML = "";
  caps.backends.forEach((value) => option(backend, value, backendLabels[value] || value, caps.backend_availability?.[value] || {}));
  selectAvailableOption(backend, [selectedBackend, "yourmt3"]);
  const ymt3 = $("#yourmt3Select"); const selectedYmt3 = ymt3.value; ymt3.innerHTML = "";
  caps.yourmt3_models.forEach((item) => option(ymt3, item.id, item.label, item));
  selectAvailableOption(ymt3, [selectedYmt3, "yptf_moe_multi_nops"]);
  const muscriptor = $("#muscriptorSelect"); const selectedMuscriptor = muscriptor.value; muscriptor.innerHTML = "";
  caps.muscriptor_models.forEach((item) => option(muscriptor, item.id, state.language === "zh_CN" ? item.label_zh : item.label_en, item));
  selectAvailableOption(muscriptor, [selectedMuscriptor, "large"]);
  const muscriptorProcessingChain = $("#muscriptorProcessingChainSelect");
  const selectedMuscriptorProcessingChain = muscriptorProcessingChain.value || "official";
  muscriptorProcessingChain.innerHTML = "";
  caps.muscriptor_processing_chains.forEach((item) => option(
    muscriptorProcessingChain,
    item.id,
    state.language === "zh_CN" ? item.label_zh : item.label_en,
  ));
  muscriptorProcessingChain.value = [...muscriptorProcessingChain.options].some(
    (item) => item.value === selectedMuscriptorProcessingChain,
  ) ? selectedMuscriptorProcessingChain : "official";
  const trackMode = $("#trackModeSelect"); trackMode.innerHTML = "";
  caps.midi_track_modes.forEach((value) => option(trackMode, value, value === "multi_track" ? t("config.track_mode.multi") : t("config.track_mode.single")));
  const tempoMode = $("#tempoModeSelect");
  const selectedTempoMode = tempoMode.value || "fixed_auto";
  tempoMode.innerHTML = "";
  caps.tempo_modes.forEach((item) => option(tempoMode, item.id, state.language === "zh_CN" ? item.label_zh : item.label_en));
  tempoMode.value = [...tempoMode.options].some((item) => item.value === selectedTempoMode) ? selectedTempoMode : "fixed_auto";
  const limits = caps.limits;
  $("#customBpm").min = limits.custom_bpm_min;
  $("#customBpm").max = limits.custom_bpm_max;
  updateConditionalControls();
}

function renderModes() {
  if (!state.capabilities) return;
  const list = $("#modeList"); list.innerHTML = "";
  const select = $("#modeSelect"); select.innerHTML = "";
  const selectedDefinition = state.capabilities.modes.find((item) => item.id === state.selectedMode);
  if (!isJobRunning() && selectedDefinition?.available === false) {
    state.selectedMode = state.capabilities.modes.find((item) => item.available !== false)?.id || "";
  }
  state.capabilities.modes.forEach((mode, index) => {
    const label = state.language === "zh_CN" ? mode.label_zh : mode.label_en;
    const option = document.createElement("option");
    option.value = mode.id;
    option.textContent = mode.available === false ? `${label} · ${t("model.unavailable")}` : label;
    option.disabled = mode.available === false;
    if (mode.unavailable_reason) option.title = mode.unavailable_reason;
    select.append(option);
    const button = document.createElement("button");
    button.type = "button"; button.className = `mode-button ${mode.id === state.selectedMode ? "is-active" : ""}`;
    button.setAttribute("aria-pressed", String(mode.id === state.selectedMode));
    button.disabled = mode.available === false;
    if (mode.unavailable_reason) button.title = mode.unavailable_reason;
    button.dataset.mode = mode.id; button.dataset.kind = mode.kind;
    const typeLabel = t(`mode.kind.${mode.kind === "separation" ? "separation" : "midi"}`);
    button.innerHTML = `<span class="mode-index">${String(index + 1).padStart(2, "0")}</span><span class="mode-label"><strong>${escapeHtml(label)}</strong><small>${typeLabel}</small></span><i class="mode-kind"></i>`;
    button.addEventListener("click", () => selectMode(mode.id));
    list.append(button);
  });
  select.value = state.selectedMode;
}
function selectMode(mode) {
  const definition = state.capabilities?.modes.find((item) => item.id === mode);
  if (definition?.available === false) {
    toast(t("error.route_unavailable", { reason: definition.unavailable_reason || t("model.unavailable") }), "error");
    return;
  }
  if (isJobRunning()) {
    toast(t("error.route_locked"), "error");
    renderModes();
    return;
  }
  if (mode !== state.selectedMode) prepareNewRequest();
  state.selectedMode = mode;
  state.progressStageSignature = "";
  renderProgressStages(mode);
  renderModes(); updateConditionalControls(); updateStartLabel(); updateReadyState();
  $("#routeDescription").textContent = t(`route.${mode}`);
}
function updateConditionalControls() {
  const isSmart = state.selectedMode === "smart";
  const backend = $("#backendSelect").value;
  $("#backendField").hidden = !isSmart;
  $("#yourmt3Field").hidden = !isSmart || backend !== "yourmt3";
  $("#muscriptorField").hidden = !isSmart || backend !== "muscriptor";
  $("#muscriptorProcessingChainField").hidden = !(
    (isSmart && backend === "muscriptor")
    || ["vocal_split", "six_stem_split"].includes(state.selectedMode)
  );
  // The desktop fixes the MIDI layout to multi-track; the web console keeps
  // the same contract instead of exposing a second layout the app hides.
  $("#trackModeField").hidden = true;
  const manualTempo = $("#tempoModeSelect").value === "fixed_manual";
  $("#manualBpmField").hidden = !manualTempo;
  $("#customBpm").disabled = !manualTempo;
  $("#routeDescription").textContent = t(`route.${state.selectedMode}`);
}
function updateStartLabel() {
  const split = ["vocal_split", "six_stem_split"].includes(state.selectedMode);
  const key = split ? "action.start_separation" : "action.start_conversion";
  $("#startButtonLabel").textContent = t(key);
}
function selectedRouteAvailability() {
  if (!state.capabilities) return { available: false, unavailable_reason: t("action.backend_disconnected") };
  const mode = state.capabilities.modes.find((item) => item.id === state.selectedMode);
  if (!mode || mode.available === false) return { available: false, unavailable_reason: mode?.unavailable_reason || t("model.unavailable") };
  if (state.selectedMode !== "smart") return { available: true, unavailable_reason: null };
  const backend = $("#backendSelect").value;
  const backendStatus = state.capabilities.backend_availability?.[backend];
  if (!backend || backendStatus?.available === false) return { available: false, unavailable_reason: backendStatus?.unavailable_reason || t("model.unavailable") };
  if (backend === "yourmt3") {
    const selected = state.capabilities.yourmt3_models.find((item) => item.id === $("#yourmt3Select").value);
    return selected || { available: false, unavailable_reason: t("model.unavailable") };
  }
  if (backend === "muscriptor") {
    const selected = state.capabilities.muscriptor_models.find((item) => item.id === $("#muscriptorSelect").value);
    return selected || { available: false, unavailable_reason: t("model.unavailable") };
  }
  return { available: true, unavailable_reason: null };
}
function updateReadyState() {
  const running = isJobRunning();
  const busy = running || state.submissionPending;
  const availability = selectedRouteAvailability();
  $("#startButton").disabled = !state.audioFile || !state.capabilities || !availability.available || busy;
  $("#stopButton").disabled = !running;
  if (state.submissionPending) {
    $("#readyText").textContent = t("action.submitting");
  } else if (running) {
    $("#readyText").textContent = state.currentJob.status === "queued" ? t("action.ready_queued") : t("action.ready_running");
  } else if (!state.capabilities) {
    $("#readyText").textContent = t("action.backend_disconnected");
  } else if (!availability.available) {
    $("#readyText").textContent = t("action.route_unavailable", { reason: availability.unavailable_reason || t("model.unavailable") });
  } else if (!state.audioFile) {
    $("#readyText").textContent = t("action.select_audio");
  } else {
    $("#readyText").textContent = `${state.audioFile.name} · ${formatBytes(state.audioFile.size)}`;
  }
  updateWorkflowGuide();
}

function setAudioFile(file) {
  if (!file) return;
  const suffix = `.${file.name.split(".").pop().toLowerCase()}`;
  const supported = state.capabilities?.limits?.audio_extensions || [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"];
  if (!supported.includes(suffix)) { toast(t("error.unsupported_format", { suffix }), "error"); return; }
  prepareNewRequest();
  if (state.sourceObjectUrl) URL.revokeObjectURL(state.sourceObjectUrl);
  state.audioFile = file; state.sourceObjectUrl = URL.createObjectURL(file);
  $("#fileInspector").hidden = false;
  $("#fileName").textContent = file.name;
  $("#fileMeta").textContent = `${formatBytes(file.size)} · ${file.type || suffix.slice(1).toUpperCase()}`;
  $("#fileInspector .file-icon").textContent = suffix.slice(1, 5).toUpperCase();
  updateReadyState();
}
function clearAudioFile() {
  prepareNewRequest();
  if (state.sourceObjectUrl) URL.revokeObjectURL(state.sourceObjectUrl);
  state.audioFile = null; state.sourceObjectUrl = ""; $("#audioInput").value = "";
  $("#fileInspector").hidden = true; updateReadyState();
}
function buildInferenceOptions() {
  const tempoMode = $("#tempoModeSelect").value || "fixed_auto";
  const bpm = tempoMode === "fixed_manual" ? Number($("#customBpm").value) : null;
  if (tempoMode === "fixed_manual" && !Number.isFinite(bpm)) throw new Error(t("error.invalid_bpm"));
  return {
    processing_mode: state.selectedMode,
    transcription_backend: $("#backendSelect").value || "yourmt3",
    yourmt3_model: $("#yourmt3Select").value || "yptf_moe_multi_nops",
    muscriptor_model: $("#muscriptorSelect").value || "large",
    muscriptor_processing_chain: $("#muscriptorProcessingChainSelect").value || "official",
    muscriptor_instruments: [],
    midi_track_mode: $("#trackModeSelect").value || "multi_track",
    tempo_mode: tempoMode,
    custom_bpm: bpm,
    use_gpu: true,
    gpu_device: 0,
    language: state.language,
  };
}

async function startPrimaryJob() {
  if (!state.audioFile || state.submissionPending || isJobRunning()) return;
  let options;
  try { options = buildInferenceOptions(); } catch (error) { toast(error.message, "error"); return; }
  const form = new FormData(); form.append("audio", state.audioFile, state.audioFile.name); form.append("options", JSON.stringify(options));
  prepareNewRequest();
  state.submissionPending = true;
  resetResult();
  updateReadyState();
  try {
    const job = await api("/api/v1/jobs", { method: "POST", body: form });
    state.submissionPending = false;
    state.currentJob = job; showProgress(job); updateReadyState(); watchJob(job.id, handlePrimaryJobUpdate);
    await refreshJobs();
  } catch (error) { toast(t("error.submit_job", { error: error.message }), "error"); }
  finally {
    state.submissionPending = false;
    updateReadyState();
  }
}
function showProgress(job, { scroll = true } = {}) {
  $("#progressPanel").hidden = false; $("#jobId").textContent = job.id; updateProgress(job);
  if (scroll) $("#progressPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function updateProgress(job) {
  const value = Math.max(0, Math.min(1, Number(job.progress?.overall_progress) || 0));
  $("#progressBar").style.width = `${value * 100}%`;
  $("#progressPercent").textContent = `${Math.round(value * 100)}%`;
  const definitions = renderProgressStages(workflowMode(job));
  const order = definitions.map(([stage]) => stage);
  const rawStage = job.status === "queued" ? "queued" : (job.progress?.stage || job.status);
  const stage = rawStage === "starting" ? "queued" : rawStage;
  $("#progressMessage").textContent = currentProgressLabel(job);
  const sourceBpm = job.progress?.source_bpm;
  const targetBpm = job.progress?.target_bpm;
  $("#progressDetail").textContent = Number.isFinite(sourceBpm) && Number.isFinite(targetBpm)
    ? t("progress.detail_bpm", { source: Number(sourceBpm).toFixed(1), target: Number(targetBpm).toFixed(1) })
    : "";
  const index = order.indexOf(stage);
  $$("[data-stage]", $("#stageRow")).forEach((node) => {
    const nodeIndex = order.indexOf(node.dataset.stage);
    node.classList.toggle("is-active", node.dataset.stage === stage || (stage === "starting" && node.dataset.stage === "queued"));
    node.classList.toggle("is-complete", index > nodeIndex);
  });
  updateWorkflowGuide();
}
function watchJob(jobId, onUpdate) {
  state.eventSources.get(jobId)?.close();
  const source = new EventSource(`${state.apiBase}/api/v1/jobs/${jobId}/events`);
  state.eventSources.set(jobId, source);
  source.addEventListener("job", (event) => {
    const job = JSON.parse(event.data); onUpdate(job);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      source.close();
      state.eventSources.delete(jobId);
    }
  });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) {
      state.eventSources.delete(jobId);
      return;
    }
    toast(t("error.event_stream", { id: jobId.slice(0, 8) }), "error");
  };
  return source;
}
function handlePrimaryJobUpdate(job) {
  state.currentJob = job; updateProgress(job); updateReadyState();
  if (job.status === "succeeded") { renderPrimaryResult(job); refreshJobs(); }
  else if (job.status === "failed") { renderFailure(job.error || t("error.backend_inference")); refreshJobs(); }
  else if (job.status === "cancelled") { renderFailure(t("error.task_cancelled"), false); refreshJobs(); }
}
async function stopCurrentJob() {
  if (!state.currentJob) return;
  try {
    state.currentJob = await api(`/api/v1/jobs/${state.currentJob.id}/cancel`, { method: "POST" });
    updateProgress(state.currentJob); updateReadyState();
  } catch (error) { toast(t("error.stop_job", { error: error.message }), "error"); }
}

function resetResult() {
  stopTransport(); state.tracks = [];
  $("#progressPanel").hidden = true;
  $("#resultPanel").hidden = true; $("#mixerPanel").hidden = true;
  $("#retryJob").hidden = true;
  $("#deleteJob").hidden = true;
  $("#resultMetrics").innerHTML = ""; $("#artifactList").innerHTML = ""; $("#trackStack").innerHTML = "";
  updateWorkflowGuide();
}
function renderFailure(message, error = true, scroll = true) {
  $("#resultPanel").hidden = false;
  $("#resultLead").textContent = error ? t("result.failed") : t("result.cancelled");
  $("#resultMetrics").innerHTML = `<div class="metric-card"><span>${escapeHtml(t("result.status"))}</span><strong class="metric-alert">${escapeHtml(error ? t("job.status.failed") : t("job.status.cancelled"))}</strong></div>`;
  const help = error ? t("result.failure_help") : t("result.cancelled_help");
  $("#artifactList").innerHTML = `<div class="artifact-row is-log"><span class="artifact-type">LOG</span><div><strong>${escapeHtml(message)}</strong><small>${escapeHtml(help)}</small></div></div>`;
  $("#retryJob").hidden = !state.currentJob || !TERMINAL_JOB_STATUSES.has(state.currentJob.status);
  $("#deleteJob").hidden = !state.currentJob || !TERMINAL_JOB_STATUSES.has(state.currentJob.status);
  updateWorkflowGuide();
  if (scroll) $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function metric(label, value) { return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`; }
function modeLabel(modeId) {
  const mode = state.capabilities?.modes.find((item) => item.id === modeId);
  if (!mode) return modeId || "—";
  return state.language === "zh_CN" ? mode.label_zh : mode.label_en;
}
function artifactLabel(kind) {
  const key = `result.artifact.${kind}`;
  return state.messages[state.language][key] ? t(key) : t("result.artifact.generic");
}
function renderArtifacts(job) {
  const warningRows = (job.result?.quality_warnings || []).map((warning) => {
    const knownKey = `result.warning.${warning}`;
    const message = state.messages[state.language][knownKey]
      ? t(knownKey)
      : t("result.warning.unknown", { code: warning });
    return `<div class="artifact-row is-warning"><span class="artifact-type">WARN</span><div><strong>${escapeHtml(message)}</strong><small>${escapeHtml(t("result.warning.help"))}</small></div></div>`;
  });
  const artifactRows = job.artifacts.map((artifact) => `
    <div class="artifact-row">
      <span class="artifact-type">${escapeHtml(artifact.name.split(".").pop().slice(0, 4).toUpperCase())}</span>
      <div><strong>${escapeHtml(artifact.name)}</strong><small>${escapeHtml(artifactLabel(artifact.kind))} · ${formatBytes(artifact.size)}</small></div>
      <a href="${escapeHtml(artifactUrl(artifact))}" download>${escapeHtml(t("result.download"))} ↓</a>
    </div>`);
  $("#artifactList").innerHTML = [...warningRows, ...artifactRows].join("");
}
function renderPrimaryResult(job, { restoreTracks = true, scroll = true } = {}) {
  $("#retryJob").hidden = true;
  $("#deleteJob").hidden = false;
  state.currentJob = job;
  const result = job.result || {};
  $("#resultPanel").hidden = false;
  $("#resultLead").textContent = result.manual_midi_required ? t("result.separation_completed") : t("result.completed");
  const metrics = [];
  metrics.push(metric(t("result.metric.mode"), modeLabel(result.mode || state.selectedMode)));
  metrics.push(metric(t("result.metric.elapsed"), t("result.seconds", { value: Number(result.processing_time || 0).toFixed(1) })));
  if (result.total_notes != null) metrics.push(metric(t("result.metric.notes"), String(result.total_notes)));
  if (result.track_count != null) metrics.push(metric(t("result.metric.tracks"), String(result.track_count)));
  if (result.beat?.bpm_display) metrics.push(metric(t("result.metric.bpm"), result.beat.bpm_display));
  $("#resultMetrics").innerHTML = metrics.join(""); renderArtifacts(job);
  if (result.manual_midi_required && restoreTracks) buildServerTracks(job);
  updateWorkflowGuide();
  if (scroll) $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function buildServerTracks(job) {
  state.tracks = [];
  const resultTracks = job.result?.tracks || [];
  resultTracks.forEach((item, index) => {
    const artifact = job.artifacts.find((entry) => entry.id === item.artifact_id);
    if (!artifact) throw new Error(t("error.missing_track_artifact", { id: item.id }));
    state.tracks.push(makeTrack({
      id: item.id, name: item.name, color: TRACK_COLORS[index % TRACK_COLORS.length],
      audioUrl: artifactUrl(artifact), fileName: artifact.name, parentJobId: job.id, serverTrackId: item.id,
    }));
  });
  try {
    const jobs = await api("/api/v1/jobs");
    state.tracks.forEach((track) => {
      const child = jobs.find((candidate) => candidate.parent_job_id === job.id && candidate.track_id === track.serverTrackId);
      if (!child) return;
      track.midiJob = child;
      track.route = child.request?.route || child.result?.route || "";
      track.midiEnabled = Boolean(track.route);
      if (child.status === "succeeded") {
        track.midiArtifact = child.artifacts.find((artifact) => artifact.kind === "midi") || null;
        track.statusKey = track.midiArtifact ? "track.midi_completed" : "track.midi_missing";
      } else if (child.status === "failed") {
        track.statusKey = "track.previous_failed";
        track.statusVars = { error: child.error || t("error.backend_inference") };
      } else if (child.status === "cancelled") {
        track.statusKey = "track.previous_cancelled";
      } else {
        track.statusKey = child.status === "queued" ? "track.queued" : "track.transcribing";
        track.statusVars = { position: child.queue_position || 1 };
      }
    });
  } catch (error) {
    toast(t("error.restore_tracks", { error: error.message }), "error");
  }
  renderMixer();
  await Promise.all(state.tracks.map(loadTrackAudio));
  redrawWaveforms();
}
function makeTrack(values) {
  return {
    id: values.id || `local-${crypto.randomUUID()}`, name: values.name || t("track.local"), color: values.color || TRACK_COLORS[state.tracks.length % TRACK_COLORS.length],
    audioUrl: values.audioUrl || "", localFile: values.localFile || null, fileName: values.fileName || values.localFile?.name || "audio",
    parentJobId: values.parentJobId || null, serverTrackId: values.serverTrackId || null, buffer: null, loading: false,
    muted: false, solo: false, gainDb: 0, offset: 0, midiEnabled: false, route: "", statusKey: "track.no_route", statusVars: {}, statusClass: "", midiJob: null, midiArtifact: null,
  };
}
async function ensureAudioContext() {
  if (!state.audioContext) state.audioContext = new AudioContext();
  if (state.audioContext.state === "suspended") await state.audioContext.resume();
  return state.audioContext;
}
async function loadTrackAudio(track) {
  if (track.buffer || track.loading) return;
  track.loading = true;
  if (!track.midiArtifact && !track.midiJob) updateTrackStatus(track, "track.waveform_decoding", {}, "is-working");
  try {
    const context = await ensureAudioContext();
    const arrayBuffer = track.localFile ? await track.localFile.arrayBuffer() : await fetch(track.audioUrl).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.arrayBuffer();
    });
    track.buffer = await context.decodeAudioData(arrayBuffer.slice(0));
    if (track.midiArtifact) updateTrackDownload(track, track.midiArtifact);
    else if (!track.midiJob) updateTrackStatus(track, "track.waveform_loaded", { time: formatTime(track.buffer.duration) });
  } catch (error) { updateTrackStatus(track, "track.waveform_failed", { error: error.message }, "is-error"); }
  finally { track.loading = false; updateTransportTime(); }
}

function renderMixer({ scroll = true } = {}) {
  $("#mixerPanel").hidden = false; const stack = $("#trackStack"); stack.innerHTML = "";
  state.tracks.forEach((track) => stack.append(createTrackRow(track)));
  state.tracks.filter((track) => track.midiArtifact).forEach((track) => updateTrackDownload(track, track.midiArtifact));
  state.tracks
    .filter((track) => track.midiJob && !["succeeded", "failed", "cancelled"].includes(track.midiJob.status))
    .forEach((track) => {
      const button = $(".convert-button", trackRow(track));
      watchJob(track.midiJob.id, (update) => applyTrackJobUpdate(track, button, update));
    });
  applyZoom(); updateTransportTime();
  if (scroll) $("#mixerPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function createTrackRow(track) {
  const row = document.createElement("article"); row.className = "track-row"; row.dataset.trackId = track.id; row.style.setProperty("--track-color", track.color);
  const routes = state.capabilities?.manual_midi_routes || [];
  const selectedRoute = routes.find((route) => route.id === track.route);
  const routeAvailable = Boolean(selectedRoute?.available);
  const routeOptions = [`<option value="">${escapeHtml(t("track.choose_route"))}</option>`, ...routes.map((route) => `<option value="${escapeHtml(route.id)}" ${route.id === track.route ? "selected" : ""} ${route.available === false ? "disabled" : ""} title="${escapeHtml(route.unavailable_reason || "")}">${escapeHtml(route.available === false ? `${route.label} · ${t("model.unavailable")}` : route.label)}</option>`)];
  const terminalMidi = track.midiJob && ["succeeded", "failed", "cancelled"].includes(track.midiJob.status);
  const activeMidi = track.midiJob && !terminalMidi;
  const convertLabel = activeMidi ? t("track.stop_conversion") : track.midiJob ? t("track.retry_conversion") : t("track.start_conversion");
  row.innerHTML = `
    <div class="track-head">
      <div class="track-name"><strong>♪ ${escapeHtml(track.name)}</strong><small>${escapeHtml(track.fileName)}</small></div>
      <button class="track-button mute ${track.muted ? "is-active" : ""}" type="button" aria-pressed="${track.muted}">${escapeHtml(t("track.mute"))}</button>
      <button class="track-button solo ${track.solo ? "is-active" : ""}" type="button" aria-pressed="${track.solo}">${escapeHtml(t("track.solo"))}</button>
      <div class="track-midi-control"><label><input class="midi-enabled" type="checkbox" ${track.midiEnabled ? "checked" : ""} ${activeMidi ? "disabled" : ""} /> ${escapeHtml(t("track.convert_to_midi"))}</label><select class="route-select" ${activeMidi ? "disabled" : ""}>${routeOptions.join("")}</select><button class="convert-button" type="button" ${track.midiEnabled && track.route && routeAvailable ? "" : "disabled"}>${escapeHtml(convertLabel)}</button></div>
      <button class="track-button remove-track" type="button">${escapeHtml(t("track.remove"))}</button>
    </div>
    <div class="waveform-scroll"><div class="waveform-content"><canvas></canvas><i class="playhead"></i></div></div>
    <div class="track-controls">
      <label><span>${escapeHtml(t("track.volume"))}</span><input class="gain" type="range" min="-60" max="0" step="0.5" value="${track.gainDb}" /><output>${track.gainDb.toFixed(1)} dB</output></label>
      <label><span>${escapeHtml(t("track.offset"))}</span><input class="offset" type="range" min="-10" max="10" step="0.01" value="${track.offset}" /><output>${track.offset >= 0 ? "+" : ""}${track.offset.toFixed(2)}s</output></label>
      <span class="track-status ${escapeHtml(track.statusClass)}">${escapeHtml(t(track.statusKey, track.statusVars))}</span>
    </div>`;
  const mute = $(".mute", row), solo = $(".solo", row), enabled = $(".midi-enabled", row), route = $(".route-select", row), convert = $(".convert-button", row);
  mute.addEventListener("click", () => { track.muted = !track.muted; mute.classList.toggle("is-active", track.muted); mute.setAttribute("aria-pressed", String(track.muted)); refreshActiveGains(); });
  solo.addEventListener("click", () => { track.solo = !track.solo; solo.classList.toggle("is-active", track.solo); solo.setAttribute("aria-pressed", String(track.solo)); refreshActiveGains(); });
  enabled.addEventListener("change", () => { track.midiEnabled = enabled.checked; const status = routes.find((item) => item.id === track.route); convert.disabled = !(track.midiEnabled && track.route && status?.available); });
  route.addEventListener("change", () => { track.route = route.value; const status = routes.find((item) => item.id === track.route); updateTrackStatus(track, track.route ? "track.route_ready" : "track.no_route"); convert.disabled = !(track.midiEnabled && track.route && status?.available); });
  convert.addEventListener("click", () => convertTrackToMidi(track, convert));
  $(".remove-track", row).addEventListener("click", () => removeTrack(track.id));
  const gain = $(".gain", row), offset = $(".offset", row);
  gain.addEventListener("input", () => { track.gainDb = Number(gain.value); gain.nextElementSibling.value = `${track.gainDb.toFixed(1)} dB`; refreshActiveGains(); });
  offset.addEventListener("input", () => { track.offset = Number(offset.value); offset.nextElementSibling.value = `${track.offset >= 0 ? "+" : ""}${track.offset.toFixed(2)}s`; updateTransportTime(); redrawWaveform(track); });
  $(".waveform-scroll", row).addEventListener("click", (event) => {
    const bounds = $(".waveform-content", row).getBoundingClientRect(); const duration = mixerDuration();
    seekTransport(((event.clientX - bounds.left) / bounds.width) * duration);
  });
  requestAnimationFrame(() => redrawWaveform(track)); return row;
}
function trackRow(track) { return $(`.track-row[data-track-id="${CSS.escape(track.id)}"]`); }
function updateTrackStatus(track, key, variables = {}, className = "") {
  track.statusKey = key; track.statusVars = variables; track.statusClass = className;
  const row = trackRow(track); if (!row) return;
  const status = $(".track-status", row); status.className = `track-status ${className}`; status.textContent = t(key, variables);
}
function updateTrackDownload(track, artifact) {
  const row = trackRow(track); if (!row) return; const status = $(".track-status", row);
  track.statusKey = "track.midi_completed"; track.statusVars = {}; track.statusClass = "is-done";
  status.className = "track-status is-done"; status.innerHTML = `${escapeHtml(t("track.midi_completed"))} · <a href="${escapeHtml(artifactUrl(artifact))}" download>${escapeHtml(artifact.name)} ↓</a>`;
}
function applyTrackJobUpdate(track, button, update) {
  track.midiJob = update;
  const controls = trackRow(track);
  const enabled = $(".midi-enabled", controls);
  const route = $(".route-select", controls);
  if (update.status === "succeeded") {
    track.midiArtifact = update.artifacts.find((artifact) => artifact.kind === "midi");
    updateTrackDownload(track, track.midiArtifact);
    button.disabled = false; button.textContent = t("track.retry_conversion"); enabled.disabled = false; route.disabled = false; refreshJobs();
  } else if (update.status === "failed") {
    updateTrackStatus(track, "track.conversion_failed", { error: update.error }, "is-error");
    button.disabled = false; button.textContent = t("track.retry_conversion"); enabled.disabled = false; route.disabled = false; refreshJobs();
  } else if (update.status === "cancelled") {
    updateTrackStatus(track, "track.conversion_cancelled", {}, "is-error");
    button.disabled = false; button.textContent = t("track.retry_conversion"); enabled.disabled = false; route.disabled = false; refreshJobs();
  } else {
    const key = update.status === "queued" ? "track.queued" : "track.transcribing";
    const variables = { position: update.queue_position || 1 };
    updateTrackStatus(track, key, variables, "is-working");
  }
}
function removeTrack(id) {
  const index = state.tracks.findIndex((track) => track.id === id); if (index < 0) return;
  state.tracks.splice(index, 1); trackRow({ id })?.remove(); updateTransportTime();
}
async function addLocalTracks(files) {
  [...files].forEach((file) => state.tracks.push(makeTrack({ name: file.name.replace(/\.[^.]+$/, ""), localFile: file, fileName: file.name })));
  renderMixer(); await Promise.all(state.tracks.filter((track) => track.localFile && !track.buffer).map(loadTrackAudio)); redrawWaveforms();
}

async function convertTrackToMidi(track, button) {
  if (track.midiJob && !["succeeded", "failed", "cancelled"].includes(track.midiJob.status)) {
    button.disabled = true;
    button.textContent = t("track.stopping");
    try {
      track.midiJob = await api(`/api/v1/jobs/${track.midiJob.id}/cancel`, { method: "POST" });
      updateTrackStatus(track, "track.stopping_conversion", {}, "is-working");
    } catch (error) {
      updateTrackStatus(track, "track.stop_failed", { error: error.message }, "is-error");
      button.disabled = false;
      button.textContent = t("track.stop_conversion");
    }
    return;
  }
  if (!track.midiEnabled || !track.route) { toast(t("track.select_route_first"), "error"); return; }
  const tempoMode = $("#tempoModeSelect").value || "fixed_auto";
  const options = { route: track.route, muscriptor_instruments: [], muscriptor_processing_chain: $("#muscriptorProcessingChainSelect").value || "official", tempo_mode: tempoMode, custom_bpm: tempoMode === "fixed_manual" ? Number($("#customBpm").value) : null, use_gpu: true, gpu_device: 0, language: state.language };
  button.disabled = true; updateTrackStatus(track, "track.submitting", {}, "is-working");
  try {
    let job;
    if (track.parentJobId && track.serverTrackId) {
      job = await api(`/api/v1/jobs/${track.parentJobId}/tracks/${encodeURIComponent(track.serverTrackId)}/midi`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(options) });
    } else if (track.localFile) {
      const form = new FormData(); form.append("audio", track.localFile, track.localFile.name); form.append("options", JSON.stringify(options));
      job = await api("/api/v1/manual-jobs", { method: "POST", body: form });
    } else throw new Error(t("track.no_source"));
    track.midiJob = job;
    button.disabled = false;
    button.textContent = t("track.stop_conversion");
    const controls = trackRow(track);
    $(".midi-enabled", controls).disabled = true;
    $(".route-select", controls).disabled = true;
    updateTrackStatus(track, job.status === "queued" ? "track.queued" : "track.transcribing", { position: job.queue_position || 1 }, "is-working");
    watchJob(job.id, (update) => applyTrackJobUpdate(track, button, update));
    refreshJobs();
  } catch (error) { updateTrackStatus(track, "track.submit_failed", { error: error.message }, "is-error"); button.disabled = false; }
}

function mixerDuration() { return Math.max(0, ...state.tracks.map((track) => track.buffer ? Math.max(0, track.offset) + track.buffer.duration : 0)); }
function dbToGain(db) { return db <= -60 ? 0 : 10 ** (db / 20); }
async function playTransport() {
  if (state.playing) return; const context = await ensureAudioContext(); const duration = mixerDuration();
  if (!duration) { toast(t("track.audio_not_decoded"), "error"); return; }
  if (state.playhead >= duration) state.playhead = 0;
  state.playing = true; state.playStartedAt = context.currentTime - state.playhead; state.activeSources = [];
  const anySolo = state.tracks.some((track) => track.solo);
  state.tracks.forEach((track) => {
    if (!track.buffer) return;
    const localTime = state.playhead - track.offset; const source = context.createBufferSource(); const gain = context.createGain();
    source.buffer = track.buffer; gain.gain.value = (track.muted || (anySolo && !track.solo)) ? 0 : dbToGain(track.gainDb);
    source.connect(gain).connect(context.destination);
    if (localTime < 0) source.start(context.currentTime - localTime, 0);
    else if (localTime < track.buffer.duration) source.start(context.currentTime, localTime);
    else return;
    state.activeSources.push({ source, gain, track });
  });
  transportFrame();
}
function pauseTransport() {
  if (!state.playing) return; const context = state.audioContext; state.playhead = Math.max(0, context.currentTime - state.playStartedAt); stopSources(); state.playing = false; cancelAnimationFrame(state.raf); updateTransportTime();
}
function stopTransport() { stopSources(); state.playing = false; state.playhead = 0; cancelAnimationFrame(state.raf); updateTransportTime(); }
function stopSources() { state.activeSources.forEach(({ source }) => { try { source.stop(); } catch { /* already ended */ } }); state.activeSources = []; }
function seekTransport(seconds) { const wasPlaying = state.playing; pauseTransport(); state.playhead = Math.max(0, Math.min(mixerDuration(), seconds)); updateTransportTime(); if (wasPlaying) playTransport(); }
function transportFrame() {
  if (!state.playing) return; state.playhead = state.audioContext.currentTime - state.playStartedAt;
  if (state.playhead >= mixerDuration()) { stopTransport(); return; }
  updateTransportTime(); state.raf = requestAnimationFrame(transportFrame);
}
function updateTransportTime() {
  const duration = mixerDuration(); $("#transportTime").textContent = `${formatTime(state.playhead)} / ${formatTime(duration)}`;
  state.tracks.forEach((track) => { const row = trackRow(track); if (!row) return; const playhead = $(".playhead", row); playhead.style.left = `${duration ? (state.playhead / duration) * 100 : 0}%`; });
}
function refreshActiveGains() { const anySolo = state.tracks.some((track) => track.solo); state.activeSources.forEach(({ gain, track }) => { gain.gain.value = (track.muted || (anySolo && !track.solo)) ? 0 : dbToGain(track.gainDb); }); }
function applyZoom() {
  state.zoom = Number($("#zoomSlider").value); $("#zoomValue").textContent = `${state.zoom}×`;
  $$(".waveform-content").forEach((node) => { node.style.width = `${state.zoom * 100}%`; }); redrawWaveforms();
}
function redrawWaveforms() { state.tracks.forEach(redrawWaveform); updateTransportTime(); }
function redrawWaveform(track) {
  const row = trackRow(track); if (!row) return; const canvas = $("canvas", row); const content = $(".waveform-content", row);
  const width = Math.max(1, Math.round(content.clientWidth * devicePixelRatio)); const height = Math.max(1, Math.round(content.clientHeight * devicePixelRatio));
  if (canvas.width !== width) canvas.width = width; if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, width, height); ctx.strokeStyle = track.color; ctx.fillStyle = `${track.color}18`; ctx.lineWidth = Math.max(1, devicePixelRatio);
  ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
  if (!track.buffer) return; const data = track.buffer.getChannelData(0); const total = mixerDuration() || track.buffer.duration;
  const offsetX = (track.offset / total) * width; const audioWidth = (track.buffer.duration / total) * width; const samplesPerPixel = data.length / Math.max(1, audioWidth);
  ctx.beginPath();
  for (let x = 0; x < Math.max(1, audioWidth); x += 1) {
    const start = Math.floor(x * samplesPerPixel); const end = Math.min(data.length, Math.floor((x + 1) * samplesPerPixel)); let min = 1; let max = -1;
    for (let sample = start; sample < end; sample += Math.max(1, Math.floor(samplesPerPixel / 24))) { const value = data[sample]; if (value < min) min = value; if (value > max) max = value; }
    const px = offsetX + x; ctx.moveTo(px, ((1 + min) * height) / 2); ctx.lineTo(px, ((1 + max) * height) / 2);
  }
  ctx.stroke();
}

function openJob(job, { scroll = true } = {}) {
  state.currentJob = job;
  const mode = job.request?.processing_mode || job.result?.mode;
  if (mode && state.capabilities?.modes.some((item) => item.id === mode)) {
    state.selectedMode = mode;
    renderModes();
    updateConditionalControls();
    updateStartLabel();
  }
  showProgress(job, { scroll });
  updateReadyState();
  if (job.status === "succeeded") renderPrimaryResult(job);
  else if (job.status === "failed") renderFailure(job.error || t("error.backend_inference"));
  else if (job.status === "cancelled") renderFailure(t("error.task_cancelled"), false);
  else watchJob(job.id, handlePrimaryJobUpdate);
}

async function refreshJobs() {
  if (!state.capabilities) return [];
  try {
    const jobs = await api("/api/v1/jobs"); const root = $("#recentJobs"); root.innerHTML = "";
    if (!jobs.length) { root.innerHTML = `<p class="empty-copy">${escapeHtml(t("jobs.empty"))}</p>`; return jobs; }
    jobs.slice(0, 5).forEach((job) => {
      const item = document.createElement("button"); item.type = "button"; item.className = `recent-job is-${job.status}`;
      item.innerHTML = `<i></i><div><strong>${escapeHtml(job.original_filename)}</strong><span>${escapeHtml(t(`job.status.${job.status}`))} · ${job.id.slice(0, 8)}</span></div>`;
      item.addEventListener("click", () => openJob(job));
      root.append(item);
    });
    return jobs;
  } catch (error) { toast(t("error.read_jobs", { error: error.message }), "error"); return []; }
}

async function retryCurrentJob() {
  if (!state.currentJob || !TERMINAL_JOB_STATUSES.has(state.currentJob.status)) return;
  const previousId = state.currentJob.id;
  $("#retryJob").disabled = true;
  try {
    const job = await api(`/api/v1/jobs/${previousId}/retry`, { method: "POST" });
    state.currentJob = job;
    resetResult();
    showProgress(job);
    updateReadyState();
    watchJob(job.id, handlePrimaryJobUpdate);
    await refreshJobs();
  } catch (error) {
    toast(t("error.retry_job", { error: error.message }), "error");
  } finally {
    $("#retryJob").disabled = false;
  }
}

async function deleteCurrentJob() {
  if (!state.currentJob || !TERMINAL_JOB_STATUSES.has(state.currentJob.status)) return;
  if (!window.confirm(t("action.delete_job_confirm"))) return;
  const jobId = state.currentJob.id;
  $("#deleteJob").disabled = true;
  try {
    await api(`/api/v1/jobs/${jobId}?cascade=true`, { method: "DELETE" });
    state.currentJob = null;
    $("#progressPanel").hidden = true;
    resetResult();
    updateReadyState();
    await refreshJobs();
    toast(t("action.delete_job_done"));
  } catch (error) {
    toast(t("error.delete_job", { error: error.message }), "error");
  } finally {
    $("#deleteJob").disabled = false;
  }
}

function bindEvents() {
  $("#browseButton").addEventListener("click", () => $("#audioInput").click());
  $("#audioInput").addEventListener("change", (event) => setAudioFile(event.target.files[0]));
  $("#clearFile").addEventListener("click", (event) => { event.stopPropagation(); clearAudioFile(); });
  const drop = $("#dropZone");
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("is-dragging"); }));
  drop.addEventListener("drop", (event) => setAudioFile(event.dataTransfer.files[0]));
  drop.addEventListener("click", (event) => { if (!event.target.closest("button, input")) $("#audioInput").click(); });
  drop.addEventListener("keydown", (event) => {
    if (["Enter", " "].includes(event.key)) { event.preventDefault(); $("#audioInput").click(); }
  });
  $("#modeSelect").addEventListener("change", (event) => selectMode(event.target.value));
  $("#backendSelect").addEventListener("change", () => { updateConditionalControls(); updateWorkflowGuide(); });
  $("#tempoModeSelect").addEventListener("change", () => { updateConditionalControls(); updateWorkflowGuide(); });
  $("#customBpm").addEventListener("input", updateWorkflowGuide);
  $("#startButton").addEventListener("click", startPrimaryJob); $("#stopButton").addEventListener("click", stopCurrentJob);
  $("#guideAction").addEventListener("click", performGuideAction);
  $$('[data-guide-target]').forEach((button) => button.addEventListener("click", () => scrollToWorkflowTarget(button.dataset.guideTarget)));
  $("#refreshJobs").addEventListener("click", refreshJobs);
  $("#retryJob").addEventListener("click", retryCurrentJob);
  $("#deleteJob").addEventListener("click", deleteCurrentJob);
  $("#connectionButton").addEventListener("click", openApiDialog);
  $("#openApiSettings").addEventListener("click", openApiDialog);
  ["#apiProtocol", "#apiHost", "#apiPort"].forEach((selector) => {
    $(selector).addEventListener("input", updateEffectiveApiUrl);
  });
  $("#testApiConnection").addEventListener("click", testDialogConnection);
  $("#apiDialog").addEventListener("close", async () => {
    if ($("#apiDialog").returnValue !== "default") return;
    try {
      state.apiBase = apiBaseFromDialog();
    } catch (error) {
      toast(t("error.backend_connection", { error: error.message }), "error");
      return;
    }
    localStorage.setItem("musicToMidiApiBase", state.apiBase); await connectBackend();
  });
  $("#languageButton").addEventListener("click", () => { state.language = state.language === "zh_CN" ? "en_US" : "zh_CN"; localStorage.setItem("musicToMidiLanguage", state.language); applyLanguage(); });
  $("#playAll").addEventListener("click", playTransport); $("#pauseAll").addEventListener("click", pauseTransport); $("#restartAll").addEventListener("click", () => { const wasPlaying = state.playing; stopTransport(); if (wasPlaying) playTransport(); });
  $("#zoomSlider").addEventListener("input", applyZoom); $("#fitTracks").addEventListener("click", () => { $("#zoomSlider").value = "1"; applyZoom(); });
  $("#alignTracks").addEventListener("click", () => { state.tracks.forEach((track) => { track.offset = 0; const row = trackRow(track); if (row) { $(".offset", row).value = "0"; $(".offset", row).nextElementSibling.value = "+0.00s"; } }); redrawWaveforms(); });
  $("#addTrack").addEventListener("click", () => $("#addTrackInput").click());
  $("#addTrackInput").addEventListener("change", (event) => { addLocalTracks(event.target.files); event.target.value = ""; });
  window.addEventListener("resize", redrawWaveforms);
}

async function initialize() {
  try {
    await loadLocaleCatalogs();
    await loadFrontendRuntimeConfig();
    bindEvents();
    applyLanguage({ rerender: false });
    $("#routeDescription").textContent = t(`route.${state.selectedMode}`);
    document.documentElement.dataset.appState = "ready";
    await connectBackend();
    window.setInterval(probeBackend, BACKEND_HEARTBEAT_INTERVAL_MS);
  } catch (error) {
    document.documentElement.dataset.appState = "failed";
    const fatal = $("#fatalError");
    fatal.hidden = false;
    fatal.textContent = `Locale initialization failed: ${error.message}`;
    throw error;
  }
}

initialize();
