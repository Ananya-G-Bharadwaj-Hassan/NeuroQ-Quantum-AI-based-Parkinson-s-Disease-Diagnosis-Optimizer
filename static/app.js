const API_BASE = "";

let doctor = null;
let fieldMeta = null;
let attachedImage = null;
let selectedRole = "doctor";

const KEY_SYMPTOMS = ["Tremor", "Rigidity", "Bradykinesia", "PosturalInstability", "SpeechProblems"];
const FIELD_LAYOUT = [
  { title: "Demographics", fields: [
    { name: "Age", type: "number" },
    { name: "Gender", type: "select", options: [{value:0,label:"Male"},{value:1,label:"Female"}] },
    { name: "Ethnicity", type: "select", options: [{value:0,label:"Caucasian"},{value:1,label:"African American"},{value:2,label:"Asian"},{value:3,label:"Other"}] },
    { name: "EducationLevel", type: "select", options: [{value:0,label:"None"},{value:1,label:"High school"},{value:2,label:"Bachelor's"},{value:3,label:"Higher"}] },
    { name: "BMI", type: "number" },
  ]},
  { title: "Lifestyle & history", fields: [
    { name: "Smoking", type: "toggle" },
    { name: "AlcoholConsumption", type: "number" },
    { name: "PhysicalActivity", type: "number" },
    { name: "DietQuality", type: "range" },
    { name: "SleepQuality", type: "range" },
    { name: "FamilyHistoryParkinsons", type: "toggle" },
    { name: "TraumaticBrainInjury", type: "toggle" },
  ]},
  { title: "Vitals & labs", fields: [
    { name: "Hypertension", type: "toggle" },
    { name: "Diabetes", type: "toggle" },
    { name: "Depression", type: "toggle" },
    { name: "Stroke", type: "toggle" },
    { name: "SystolicBP", type: "number" },
    { name: "DiastolicBP", type: "number" },
    { name: "CholesterolTotal", type: "number" },
    { name: "CholesterolLDL", type: "number" },
    { name: "CholesterolHDL", type: "number" },
    { name: "CholesterolTriglycerides", type: "number" },
  ]},
  { title: "Cognitive & motor assessment", fields: [
    { name: "UPDRS", type: "number", hint: "Unified Parkinson's Disease Rating Scale (Key Quantum Biomarker)" },
    { name: "MoCA", type: "number", hint: "Montreal Cognitive Assessment — lower is worse (Key Quantum Biomarker)" },
    { name: "FunctionalAssessment", type: "range", hint: "Higher is better function (Key Quantum Biomarker)" },
    { name: "Tremor", type: "toggle" },
    { name: "Rigidity", type: "toggle" },
    { name: "Bradykinesia", type: "toggle" },
    { name: "PosturalInstability", type: "toggle" },
    { name: "SpeechProblems", type: "toggle" },
    { name: "SleepDisorders", type: "toggle" },
    { name: "Constipation", type: "toggle" },
  ]},
];

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("tab-signin").addEventListener("click", () => switchAuthTab("signin"));
  document.getElementById("tab-register").addEventListener("click", () => switchAuthTab("register"));
  
  const roleDocBtn = document.getElementById("role-doctor-btn");
  const rolePatBtn = document.getElementById("role-patient-btn");
  if (roleDocBtn) roleDocBtn.addEventListener("click", () => switchRole("doctor"));
  if (rolePatBtn) rolePatBtn.addEventListener("click", () => switchRole("patient"));

  document.getElementById("signin-form").addEventListener("submit", handleLogin);
  document.getElementById("register-form").addEventListener("submit", handleRegister);
  document.getElementById("logout-btn").addEventListener("click", handleLogout);
  document.getElementById("run-btn").addEventListener("click", runDiagnosis);
  document.getElementById("mri-weight").addEventListener("input", e => {
    document.getElementById("mri-weight-val").textContent = parseFloat(e.target.value).toFixed(2);
  });
  document.getElementById("quantum-weight").addEventListener("input", e => {
    document.getElementById("quantum-weight-val").textContent = parseFloat(e.target.value).toFixed(2);
  });
  setupUploadZone();
  setupBenchmarksModal();
  checkExistingSession();
});

function switchRole(role) {
  selectedRole = role;
  const isDoctor = role === "doctor";
  
  const docBtn = document.getElementById("role-doctor-btn");
  const patBtn = document.getElementById("role-patient-btn");
  if (docBtn && patBtn) {
    docBtn.classList.toggle("active", isDoctor);
    patBtn.classList.toggle("active", !isDoctor);
  }

  // Update signin form text
  const signinHeading = document.getElementById("signin-heading");
  const signinSub = document.getElementById("signin-sub");
  const signinBtn = document.getElementById("signin-btn");
  if (signinHeading) signinHeading.textContent = isDoctor ? "Doctor sign in" : "Patient sign in";
  if (signinSub) signinSub.textContent = isDoctor ? "Sign in to access physician clinical tools." : "Sign in to view your diagnostic health assessment.";
  if (signinBtn) signinBtn.textContent = isDoctor ? "Sign in as Doctor" : "Sign in as Patient";

  // Update register form text
  const regHeading = document.getElementById("register-heading");
  const regSub = document.getElementById("register-sub");
  const regNameLabel = document.getElementById("reg-name-label");
  const regNameInput = document.getElementById("reg-name");
  const regHospitalLabel = document.getElementById("reg-hospital-label");
  const regBtn = document.getElementById("register-btn");

  if (regHeading) regHeading.textContent = isDoctor ? "Create a doctor account" : "Create a patient account";
  if (regSub) regSub.textContent = isDoctor ? "Credentials are saved in MongoDB Atlas." : "Register to access your personalized screening portal.";
  if (regNameLabel) regNameLabel.textContent = isDoctor ? "Full Name (e.g. Dr. Jane Smith)" : "Patient Full Name";
  if (regNameInput) regNameInput.placeholder = isDoctor ? "Dr. Jane Smith" : "John Doe";
  if (regHospitalLabel) regHospitalLabel.innerHTML = isDoctor ? 'Hospital / Clinic Affiliation <span style="text-transform:none; opacity:0.7;">(optional)</span>' : 'Treating Hospital / Clinic <span style="text-transform:none; opacity:0.7;">(optional)</span>';
  if (regBtn) regBtn.textContent = isDoctor ? "Create Doctor Account" : "Create Patient Account";
}

function switchAuthTab(which) {
  const signin = which === "signin";
  document.getElementById("tab-signin").classList.toggle("active", signin);
  document.getElementById("tab-register").classList.toggle("active", !signin);
  document.getElementById("signin-form").classList.toggle("hidden", !signin);
  document.getElementById("register-form").classList.toggle("hidden", signin);
  document.getElementById("login-error").classList.add("hidden");
  document.getElementById("register-error").classList.add("hidden");
}

async function checkExistingSession() {
  try {
    const res = await fetch(`${API_BASE}/api/me`, { credentials: "same-origin" });
    if (!res.ok) return;
    const data = await res.json();
    const activeUser = data.user || data.doctor;
    if (activeUser) {
      doctor = activeUser;
      await enterDashboard();
    }
  } catch { /* backend offline or not ready yet */ }
}

async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;
  const errBox = document.getElementById("login-error");
  errBox.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/login`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`Authentication server error (${res.status}).`);
    }
    if (!res.ok) throw new Error(data.error || "Incorrect email or password.");
    doctor = data.user || data.doctor;
    await enterDashboard();
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const name = document.getElementById("reg-name").value.trim();
  const email = document.getElementById("reg-email").value.trim();
  const hospital = document.getElementById("reg-hospital").value.trim();
  const password = document.getElementById("reg-password").value;
  const errBox = document.getElementById("register-error");
  errBox.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/register`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, hospital, password, role: selectedRole }),
    });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`Server registration error (${res.status}).`);
    }
    if (!res.ok) throw new Error(data.error || "Couldn't create that account in MongoDB.");
    doctor = data.user || data.doctor;
    await enterDashboard();
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
}

async function handleLogout() {
  try { await fetch(`${API_BASE}/api/logout`, { method: "POST", credentials: "same-origin" }); } catch {}
  doctor = null;
  document.getElementById("view-dashboard").classList.add("hidden");
  document.getElementById("view-login").classList.remove("hidden");
  document.getElementById("signin-form").reset();
  document.getElementById("register-form").reset();
  switchAuthTab("signin");
}

async function enterDashboard() {
  const isDoctor = (doctor.role || "doctor") === "doctor";
  
  document.getElementById("doctor-name").textContent = doctor.name;
  
  const roleBadge = document.getElementById("user-role-badge");
  if (roleBadge) {
    roleBadge.textContent = isDoctor ? "Attending Physician" : "Registered Patient";
  }

  const avatar = document.getElementById("doctor-avatar");
  if (avatar) {
    if (isDoctor) {
      avatar.textContent = doctor.name.split(" ").filter(w => /[A-Za-z]/.test(w[0] || "")).map(w => w[0].toUpperCase()).slice(0, 2).join("") || "DR";
    } else {
      avatar.textContent = "PT";
    }
  }

  const panelTitle = document.getElementById("panel-title-text");
  if (panelTitle) {
    panelTitle.textContent = isDoctor ? "New Patient Clinical Assessment" : "Personal Health Symptom Assessment";
  }

  document.getElementById("view-login").classList.add("hidden");
  document.getElementById("view-dashboard").classList.remove("hidden");

  await loadFeatureSchema();
  renderForm();
}


async function loadFeatureSchema() {
  const errBox = document.getElementById("form-error");
  errBox.classList.add("hidden");
  try {
    const res = await fetch(`${API_BASE}/api/features`, { credentials: "same-origin" });
    if (!res.ok) throw new Error((await res.json()).error || "Couldn't load clinical model schema.");
    const data = await res.json();
    fieldMeta = Object.fromEntries(data.features.map(f => [f.name, f]));
    
    const qFeatures = data.quantum_features && data.quantum_features.length
      ? ` | Quantum State: [${data.quantum_features.join(", ")}]`
      : "";
    
    document.getElementById("hybrid-model-names").textContent =
      `Hybrid Stacking: Fuses ${data.classical_model.toUpperCase()} (classical) with ${data.quantum_model.toUpperCase()} (quantum). Slider adjusts interactive clinical blending weight (0.0 = Classical only, 0.20 = Learned Stacking Optimum, 1.0 = Quantum only)${qFeatures}.`;
  } catch (err) {
    fieldMeta = null;
    errBox.textContent = err.message + " (Run train_diagnosis.py to train models, then reload.)";
    errBox.classList.remove("hidden");
  }
}

function renderForm() {
  const container = document.getElementById("form-sections");
  container.innerHTML = "";
  if (!fieldMeta) return;

  const placed = new Set();
  FIELD_LAYOUT.forEach((section, idx) => {
    const present = section.fields.filter(f => fieldMeta[f.name]);
    if (!present.length) return;
    present.forEach(f => placed.add(f.name));
    container.appendChild(buildSectionCard(section.title, present, idx === 0 || idx === 3));
  });

  const leftover = Object.keys(fieldMeta).filter(n => !placed.has(n)).map(n => ({ name: n, type: "number" }));
  if (leftover.length) container.appendChild(buildSectionCard("Other clinical data", leftover, false));
}

function buildSectionCard(title, fields, openByDefault) {
  const card = document.createElement("div");
  card.className = "section-card" + (openByDefault ? " open" : "");
  const header = document.createElement("div");
  header.className = "section-header";
  header.innerHTML = `<h3>${title}</h3><span class="chevron">&#9662;</span>`;
  header.addEventListener("click", () => card.classList.toggle("open"));
  const body = document.createElement("div");
  body.className = "section-body";
  fields.forEach(f => body.appendChild(renderField(f)));
  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function renderField(layoutField) {
  const range = fieldMeta[layoutField.name] || { min: 0, max: 1 };
  const mid = Math.round(((range.min + range.max) / 2) * 100) / 100;
  const wrap = document.createElement("div");
  wrap.dataset.fieldName = layoutField.name;

  if (layoutField.type === "toggle") {
    wrap.className = "field-toggle";
    wrap.innerHTML = `
      <span>${layoutField.name}</span>
      <label class="switch">
        <input type="checkbox" data-field="${layoutField.name}" />
        <span class="track"></span>
      </label>`;
  } else if (layoutField.type === "select") {
    wrap.className = "field-group";
    const opts = layoutField.options.map(o => `<option value="${o.value}">${o.label}</option>`).join("");
    wrap.innerHTML = `<label>${layoutField.name}</label><select data-field="${layoutField.name}">${opts}</select>`;
  } else if (layoutField.type === "range") {
    wrap.className = "range-field";
    wrap.innerHTML = `
      <div class="range-label"><span>${layoutField.name}</span><span class="val" data-valfor="${layoutField.name}">${mid}</span></div>
      <input type="range" data-field="${layoutField.name}" min="${range.min}" max="${range.max}" step="0.1" value="${mid}" />
      ${layoutField.hint ? `<div class="field-hint">${layoutField.hint}</div>` : ""}`;
    wrap.querySelector("input").addEventListener("input", e => {
      wrap.querySelector(`[data-valfor="${layoutField.name}"]`).textContent = e.target.value;
    });
  } else {
    wrap.className = "field-group";
    wrap.innerHTML = `
      <label>${layoutField.name}</label>
      <input type="number" data-field="${layoutField.name}" min="${range.min}" max="${range.max}" step="0.1" value="${mid}" />
      ${layoutField.hint ? `<div class="field-hint">${layoutField.hint}</div>` : ""}`;
  }
  return wrap;
}

function collectFeatures() {
  const features = {};
  document.querySelectorAll("[data-field]").forEach(el => {
    const name = el.dataset.field;
    features[name] = el.type === "checkbox" ? (el.checked ? 1 : 0) : parseFloat(el.value);
  });
  return features;
}

function setupUploadZone() {
  const zone = document.getElementById("upload-zone");
  const input = document.getElementById("mri-input");
  const label = document.getElementById("upload-label");
  const weightWrap = document.getElementById("mri-weight-wrap");

  const attach = file => {
    attachedImage = file;
    zone.classList.add("has-file");
    label.textContent = `Attached: ${file.name}`;
    weightWrap.classList.remove("hidden");
  };

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files.length) attach(input.files[0]); });
  ["dragover", "dragenter"].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", e => { if (e.dataTransfer.files.length) attach(e.dataTransfer.files[0]); });
}

function setupBenchmarksModal() {
  const modal = document.getElementById("benchmark-modal");
  const openBtn = document.getElementById("benchmarks-btn");
  const closeBtn = document.getElementById("benchmark-close");

  if (!openBtn || !modal) return;

  openBtn.addEventListener("click", async () => {
    modal.classList.remove("hidden");
    await loadBenchmarksData();
  });

  closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", e => {
    if (e.target === modal) modal.classList.add("hidden");
  });
}

async function loadBenchmarksData() {
  const container = document.getElementById("benchmark-content");
  try {
    const res = await fetch(`${API_BASE}/api/benchmarks`, { credentials: "same-origin" });
    const data = await res.json();
    renderBenchmarkTables(data, container);
  } catch (err) {
    container.innerHTML = `<div style="color:var(--risk-high); padding:1rem;">Failed to load benchmark telemetry: ${err.message}</div>`;
  }
}

function renderBenchmarkTables(data, container) {
  let html = `
    <div style="margin-bottom:1.2rem; font-size:0.86rem; color:var(--text-muted); line-height:1.5;">
      Below is the empirical test-set evaluation comparing standard Classical baselines,
      Quantum Circuits (with supervised biomarker state encoding), and the combined
      <strong>Hybrid Quantum-Classical Stacking Classifier</strong>.
    </div>
  `;

  if (data.diagnosis && data.diagnosis.models) {
    html += `
      <div class="benchmark-table-wrap">
        <h3>Clinical Tabular Diagnosis Benchmark (N = ${data.diagnosis.n_samples})</h3>
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Category</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            ${renderModelRows(data.diagnosis.models)}
          </tbody>
        </table>
      </div>
    `;
  }

  if (data.imaging && data.imaging.models) {
    html += `
      <div class="benchmark-table-wrap">
        <h3>MRI Scan Radiomics Diagnosis Benchmark (N = ${data.imaging.n_samples})</h3>
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Category</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            ${renderModelRows(data.imaging.models)}
          </tbody>
        </table>
      </div>
    `;
  }

  html += `
    <div style="background:var(--bg-field); border:1px solid var(--border); border-radius:8px; padding:0.9rem 1.1rem; font-size:0.78rem; color:var(--text-muted); line-height:1.6;">
      <strong style="color:var(--accent);">Quantum Advantage in the Hybrid Model:</strong>
      Classical trees (XGBoost/RF) analyze high-dimensional clinical tables efficiently, while the Quantum Kernel projects key severity markers (UPDRS, MoCA, Tremor, Rigidity) into an entangled Hilbert state space. The Hybrid Stacking Meta-Learner fuses both perspectives, achieving superior AUC and robust diagnostic confidence.
    </div>
  `;

  container.innerHTML = html;
}

function renderModelRows(models) {
  const sorted = Object.entries(models).sort((a, b) => (b[1].accuracy || 0) - (a[1].accuracy || 0));
  return sorted.map(([name, m]) => {
    const isHybrid = name.includes("hybrid");
    const isQuantum = name.includes("quantum");
    const cat = isHybrid ? "Hybrid Stacking" : isQuantum ? "Quantum Circuit" : "Classical Baseline";
    const badgeClass = isHybrid ? "hybrid" : isQuantum ? "quantum" : "classical";
    const auc = m.roc && m.roc.auc ? m.roc.auc.toFixed(4) : "N/A";
    const rowClass = isHybrid ? "highlight" : "";

    return `
      <tr class="${rowClass}">
        <td><strong>${formatModelName(name)}</strong></td>
        <td><span class="badge-tag ${badgeClass}">${cat}</span></td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.precision * 100).toFixed(2)}%</td>
        <td>${(m.recall * 100).toFixed(2)}%</td>
        <td>${(m.f1 * 100).toFixed(2)}%</td>
        <td>${auc}</td>
      </tr>
    `;
  }).join("");
}

function formatModelName(n) {
  const map = {
    "hybrid_model": "Hybrid Quantum-Classical Classifier",
    "xgboost": "XGBoost Classifier",
    "random_forest": "Random Forest Ensemble",
    "logistic_regression": "Logistic Regression",
    "quantum_kernel_svm": "Quantum Kernel SVM (Angle Embedding)",
    "variational_quantum_classifier": "Variational Quantum Classifier (VQC)"
  };
  return map[n] || n.replace(/_/g, " ").toUpperCase();
}

async function runDiagnosis() {
  const errBox = document.getElementById("form-error");
  errBox.classList.add("hidden");
  if (!fieldMeta) { errBox.textContent = "Clinical model isn't loaded — reload the page."; errBox.classList.remove("hidden"); return; }

  const features = collectFeatures();

  document.getElementById("results-empty").classList.add("hidden");
  document.getElementById("results-content").classList.add("hidden");
  document.getElementById("results-loading").classList.remove("hidden");

  const loadingSteps = attachedImage
    ? ["Encoding clinical features...", "Simulating quantum kernel state...", "Reading brain MRI radiomics...", "Computing hybrid meta-fusion..."]
    : ["Encoding symptoms into Hilbert space...", "Evaluating quantum state overlaps...", "Running XGBoost classical trees...", "Synthesizing hybrid decision score..."];
  let stepIdx = 0;
  document.getElementById("loading-status").textContent = loadingSteps[0];
  const stepTimer = setInterval(() => {
    stepIdx = (stepIdx + 1) % loadingSteps.length;
    document.getElementById("loading-status").textContent = loadingSteps[stepIdx];
  }, 450);

  const fd = new FormData();
  Object.entries(features).forEach(([k, v]) => fd.append(k, v));
  fd.append("quantum_weight", document.getElementById("quantum-weight").value);
  if (attachedImage) {
    fd.append("mri", attachedImage);
    fd.append("mri_weight", document.getElementById("mri-weight").value);
  }

  let result, errorMsg = null;
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      credentials: "same-origin",
      body: fd
    });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`Server returned response status ${res.status} (${res.statusText || "Error"}).`);
    }
    if (!res.ok) throw new Error(data.error || "Prediction processing failed.");
    result = data;
  } catch (err) {
    errorMsg = err.message || "Failed to communicate with prediction backend.";
  }

  clearInterval(stepTimer);
  document.getElementById("results-loading").classList.add("hidden");

  if (errorMsg) {
    document.getElementById("results-empty").classList.remove("hidden");
    errBox.textContent = errorMsg;
    errBox.classList.remove("hidden");
    return;
  }
  renderResults(result, features);
}

function hybridBanner(label, h, cls) {
  if (!h) return "";
  const pClass = Math.round((h.p_classical || 0) * 100);
  const pQuant = Math.round((h.p_quantum || 0) * 100);
  const pHyb = Math.round((h.p_hybrid || 0) * 100);
  const cName = formatModelName(h.classical_model_name || "classical");
  const qName = formatModelName(h.quantum_model_name || "quantum");
  const strategy = h.strategy || "Stacking Meta-Learner";
  const confBand = h.confidence_band || "Moderate";

  let qubitChipsHtml = "";
  if (h.quantum_features && h.quantum_features.length) {
    qubitChipsHtml = `
      <div style="margin-top:0.6rem;">
        <span style="font-size:0.72rem; color:var(--text-faint); text-transform:uppercase; letter-spacing:0.04em;">Entangled Qubit Features:</span>
        <div class="qubit-chips">
          ${h.quantum_features.map((f, i) => `<span class="qubit-chip"><span class="q-idx">|q${i}⟩</span> ${escapeHtml(f)}</span>`).join("")}
        </div>
      </div>
    `;
  }

  return `
    <div class="hybrid-breakdown-card">
      <div class="card-title">
        <span>${escapeHtml(label)}</span>
        <span class="badge-tag hybrid">${escapeHtml(strategy)}</span>
      </div>

      <div class="model-contrib-row">
        <span class="name"><span class="badge-tag classical">Classical</span> ${escapeHtml(cName)}</span>
        <span class="val">${pClass}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill classical" style="width:${pClass}%"></div></div>

      <div class="model-contrib-row">
        <span class="name"><span class="badge-tag quantum">Quantum</span> ${escapeHtml(qName)}</span>
        <span class="val">${pQuant}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill quantum" style="width:${pQuant}%"></div></div>

      <div class="model-contrib-row" style="margin-top:0.6rem; padding-top:0.4rem; border-top:1px dashed var(--border-soft);">
        <span class="name" style="color:var(--accent); font-weight:600;">Hybrid Fused Posterior</span>
        <span class="val" style="color:var(--accent);">${pHyb}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill hybrid" style="width:${pHyb}%"></div></div>

      <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-muted); margin-top:0.4rem;">
        <span>Diagnostic Confidence: <strong>${escapeHtml(confBand)}</strong></span>
        <span>Decision Margin: <strong>${(h.confidence ? (h.confidence * 100).toFixed(0) : 50)}%</strong></span>
      </div>

      ${qubitChipsHtml}
    </div>
  `;
}

function renderResults(result, features) {
  const pct = Math.round(result.p_final * 100);
  const band = result.p_final >= 0.66 ? "high" : result.p_final >= 0.4 ? "moderate" : "low";
  const circumference = 2 * Math.PI * 70;
  const offset = circumference - (pct / 100) * circumference;
  const bandColor = band === "high" ? "#f2665b" : band === "moderate" ? "#f4b860" : "#5fd98a";
  const verdict = result.label === "Parkinson's likely" ? "Likely Parkinson's Indicators Present" : "Low Likelihood of Parkinson's";

  const clinicalBanner = hybridBanner("Clinical Hybrid (Classical + Quantum)", result.clinical, "real");
  const imagingBanner = result.mri_used
    ? hybridBanner("MRI Imaging Hybrid (Classical + Quantum)", result.imaging, "imaging") +
      `<div class="field-hint" style="margin-bottom:1rem; font-size:0.78rem;">MRI Modality Weighted <strong>${result.mri_weight.toFixed(2)}</strong> into the combined <strong>${pct}%</strong> overall diagnosis.</div>`
    : `<div class="mode-banner imaging">No brain MRI attached — this assessment uses the Clinical Hybrid model only.</div>`;

  const flags = KEY_SYMPTOMS.filter(s => features[s] === 1);
  const flagsHtml = flags.length
    ? `<ul class="flags-list">${flags.map(f => `<li><span class="flag-dot"></span>${escapeHtml(f)} reported</li>`).join("")}</ul>`
    : `<p style="font-size:0.85rem; color:var(--text-muted); margin: 0 0 1rem;">No major motor symptom flags reported.</p>`;

  document.getElementById("results-content").innerHTML = `
    <div class="gauge-wrap">
      <svg width="168" height="168" viewBox="0 0 168 168">
        <circle cx="84" cy="84" r="70" fill="none" stroke="#1a2c4a" stroke-width="12"/>
        <circle cx="84" cy="84" r="70" fill="none" stroke="${bandColor}" stroke-width="12"
          stroke-linecap="round" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"/>
      </svg>
      <div class="gauge-center"><span class="pct">${pct}%</span><span class="pct-label">Probability</span></div>
    </div>
    <div class="verdict-badge ${band}">${escapeHtml(verdict)}</div>
    ${clinicalBanner}
    ${imagingBanner}
    <h3 style="font-size:0.85rem; color:var(--text-muted); margin: 0.6rem 0 0.4rem; text-transform:uppercase; letter-spacing:0.04em;">Motor symptom flags</h3>
    ${flagsHtml}
    <div class="results-footnote">
      NeuroQ Clinical Decision Support Prototype. Powered by an entangled Quantum Kernel SVM & XGBoost Stacking Classifier. Does not replace a neurologist's examination.
    </div>`;
  document.getElementById("results-content").classList.remove("hidden");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}
