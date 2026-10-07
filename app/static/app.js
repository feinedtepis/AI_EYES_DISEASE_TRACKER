/* EyeVision AI front-end (plain JS, no build step). Research prototype — not a medical device. */
"use strict";

const view = document.getElementById("view");
let META = null;
let charts = [];

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => (p == null ? "–" : `${(100 * p).toFixed(d)}%`);
const dio = (v) => (v == null ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)} D`);
const quarter = (v) => Math.round(v * 4) / 4;
const EYE = { R: "Right eye", L: "Left eye" };
const EYE_ABBR = { R: "OD", L: "OS" };

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function destroyCharts() { charts.forEach((c) => c.destroy()); charts = []; }

function disclaimer() {
  return `<div class="research-strip" role="note"><strong>Research/educational AI estimate only.</strong>
    This system is not a medical diagnostic tool and does not provide a clinical diagnosis or eyeglass prescription.
    Please consult a qualified ophthalmologist/optometrist for diagnosis and refraction.</div>`;
}

function setActive(route) {
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
}

function modelStatus() {
  const el = document.getElementById("model-status");
  if (!META) return;
  const m = META.models;
  const row = (label, info) => {
    if (!info) return `<div class="model-pill"><span class="dot none"></span>${label}: not trained</div>`;
    return `<div class="model-pill"><span class="dot ${info.is_demo ? "demo" : ""}"></span>${label}: ${esc(info.arch)}${info.is_demo ? " (synthetic demo)" : ""}</div>`;
  };
  el.innerHTML = row("Disease model", m.disease) + row("Refraction model", m.refractive) +
    `<div>Runs on ${esc(m.device)}</div>`;
}

// ------------------------------------------------------------------ router
async function route() {
  destroyCharts();
  const h = location.hash.replace(/^#/, "") || "/";
  const parts = h.split("/").filter(Boolean);
  try {
    if (!META) { META = await api("/api/meta"); modelStatus(); }
    if (parts.length === 0) { setActive("dashboard"); return await renderDashboard(); }
    if (parts[0] === "new") { setActive("new"); return renderNew(parts[1] ? decodeURIComponent(parts[1]) : ""); }
    if (parts[0] === "visit") { setActive("patients"); return await renderVisit(parts[1]); }
    if (parts[0] === "patients") { setActive("patients"); return await renderPatients(); }
    if (parts[0] === "patient") { setActive("patients"); return await renderPatient(decodeURIComponent(parts[1])); }
    if (parts[0] === "trends") { setActive("trends"); return await renderTrends(parts[1] ? decodeURIComponent(parts[1]) : ""); }
    if (parts[0] === "models") { setActive("models"); return await renderModels(); }
    view.innerHTML = `<div class="empty">Page not found. <a href="#/">Go to the dashboard</a>.</div>`;
  } catch (e) {
    view.innerHTML = `<div class="panel"><h2>Something went wrong</h2><p class="error">${esc(e.message)}</p></div>`;
  }
  view.focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);
window.addEventListener("load", route);

// ------------------------------------------------------------------ dashboard
async function renderDashboard() {
  const d = await api("/api/dashboard");
  const c = d.counts;
  if (!c.eye_examinations) {
    view.innerHTML = `<div class="page-head"><div><h1>Dashboard</h1></div></div>${disclaimer()}
      <div class="panel empty"><h2>No examinations yet</h2>
      <p>Upload a left and/or right fundus photograph to get model estimates and start a history for that patient.</p>
      <a class="btn" href="#/new">New examination</a></div>`;
    return;
  }
  const risk = d.risk_overview.map((r) => {
    const frac = r.eyes ? r.eyes_above_threshold / r.eyes : 0;
    return `<div class="prob ${r.eyes_above_threshold ? "above" : ""}">
      <div class="name">${esc(r.short)}</div>
      <div class="track" title="${r.eyes_above_threshold} of ${r.eyes} eyes"><div class="fill" style="width:${100 * frac}%"></div></div>
      <div class="val">${r.eyes_above_threshold}/${r.eyes}</div></div>`;
  }).join("");
  const recent = d.recent_visits.map((v) => `
    <tr class="click" onclick="location.hash='#/visit/${esc(v.visit_id)}'">
      <td>${esc(v.exam_date)}</td><td>${esc(v.patient_id)}</td>
      <td>${v.eyes.map((e) => EYE_ABBR[e]).join(" + ")}</td>
      <td>${v.top_class ? `${esc(v.top_class.short)} ${pct(v.top_class.probability)}` : "–"}</td>
      <td class="num">${["R", "L"].filter((e) => v.se[e] != null).map((e) => `${EYE_ABBR[e]} ${dio(quarter(v.se[e]))}`).join("<br>") || "–"}</td>
      <td>${v.demo ? '<span class="tag demo">demo model</span>' : ""}</td></tr>`).join("");
  const se = d.latest_se.map((r) => `<tr class="click" onclick="location.hash='#/trends/${encodeURIComponent(r.patient_id)}'">
      <td>${esc(r.patient_id)}</td><td>${EYE_ABBR[r.eye]}</td><td>${esc(r.exam_date)}</td>
      <td class="num">${dio(quarter(r.predicted_se))}</td><td>${r.demo ? '<span class="tag demo">demo</span>' : ""}</td></tr>`).join("");
  view.innerHTML = `
    <div class="page-head"><div><h1>Dashboard</h1><p>Overview of stored examinations and the latest model estimates.</p></div>
      <a class="btn" href="#/new">New examination</a></div>
    ${disclaimer()}
    <div class="counts">
      <div class="count"><b>${c.eye_examinations}</b><span>eye examinations</span></div>
      <div class="count"><b>${c.visits}</b><span>visits</span></div>
      <div class="count"><b>${c.patients}</b><span>patients</span></div>
    </div>
    <div class="grid-2">
      <div class="panel"><div class="panel-head"><h2>Model flags in latest examinations</h2>
        <span class="hint">eyes above the model's threshold</span></div>${risk}
        <p class="tiny muted" style="margin-top:10px">Counts the most recent examination of each eye. A flag means the model's score passed its
        validation-tuned threshold — it is not a diagnosis.</p></div>
      <div class="panel"><div class="panel-head"><h2>Latest estimated spherical equivalent</h2></div>
        <table class="data"><thead><tr><th>Patient</th><th>Eye</th><th>Date</th><th class="num">Estimate</th><th></th></tr></thead>
        <tbody>${se || '<tr><td colspan="5" class="muted">No refractive estimates.</td></tr>'}</tbody></table></div>
    </div>
    <div class="panel"><div class="panel-head"><h2>Recent examinations</h2><a href="#/patients" class="small">All patients</a></div>
      <table class="data"><thead><tr><th>Date</th><th>Patient</th><th>Eyes</th><th>Highest non-normal score</th><th class="num">Est. SE</th><th></th></tr></thead>
      <tbody>${recent}</tbody></table></div>`;
}

// ------------------------------------------------------------------ new examination
function renderNew(prefill) {
  const today = new Date().toISOString().slice(0, 10);
  view.innerHTML = `
    <div class="page-head"><div><h1>New examination</h1>
      <p>Upload colour fundus photographs. Each eye is analysed separately and saved to the patient's history.</p></div></div>
    ${disclaimer()}
    <form class="panel" id="exam-form" novalidate>
      <div class="form-grid">
        <label class="field">Patient ID<input name="patient_id" list="pids" required maxlength="40" value="${esc(prefill)}" placeholder="e.g. P-001" autocomplete="off"></label>
        <label class="field">Age<input name="age" type="number" min="0" max="120" placeholder="years"></label>
        <label class="field">Sex<select name="sex"><option value="U">Not stated</option><option value="F">Female</option><option value="M">Male</option><option value="O">Other</option></select></label>
        <label class="field">Examination date<input name="exam_date" type="date" value="${today}" max="${today}" required></label>
      </div>
      <datalist id="pids"></datalist>
      <div class="drop-grid">
        ${["R", "L"].map((e) => `
        <div class="drop" id="drop-${e}">
          <input type="file" accept="image/*" name="${e === "L" ? "left_image" : "right_image"}" aria-label="${EYE[e]} image">
          <div class="eye-label">${EYE[e]} (${EYE_ABBR[e]})</div>
          <div class="muted small">Drop a fundus photo here or click to choose</div>
        </div>`).join("")}
      </div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <button class="btn" type="submit" id="go">Analyse images</button>
        <span class="muted small">Analysis takes a few seconds per eye on a CPU.</span>
      </div>
      <div class="error" id="form-error" role="alert"></div>
    </form>`;
  api("/api/patients").then((ps) => {
    document.getElementById("pids").innerHTML = ps.map((p) => `<option value="${esc(p.patient_id)}">`).join("");
    const f = document.getElementById("exam-form");
    f.patient_id.addEventListener("change", () => {
      const p = ps.find((x) => x.patient_id === f.patient_id.value);
      if (p) { if (p.age != null) f.age.value = p.age; f.sex.value = p.sex || "U"; }
    });
  });
  ["R", "L"].forEach((e) => {
    const box = document.getElementById(`drop-${e}`);
    const inp = box.querySelector("input");
    const show = () => {
      const file = inp.files[0];
      box.querySelectorAll("img,.fname").forEach((n) => n.remove());
      if (!file) return;
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file); img.alt = `${EYE[e]} preview`;
      const nm = document.createElement("div"); nm.className = "fname tiny muted"; nm.textContent = file.name;
      box.append(img, nm);
    };
    inp.addEventListener("change", show);
    box.addEventListener("dragover", (ev) => { ev.preventDefault(); box.classList.add("drag"); });
    box.addEventListener("dragleave", () => box.classList.remove("drag"));
    box.addEventListener("drop", (ev) => { ev.preventDefault(); box.classList.remove("drag"); inp.files = ev.dataTransfer.files; show(); });
  });
  document.getElementById("exam-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target, err = document.getElementById("form-error"), btn = document.getElementById("go");
    err.textContent = "";
    if (!f.patient_id.value.trim()) { err.textContent = "Enter a patient ID."; return; }
    if (!f.left_image.files.length && !f.right_image.files.length) { err.textContent = "Add at least one eye image."; return; }
    const fd = new FormData();
    fd.append("patient_id", f.patient_id.value.trim());
    if (f.age.value) fd.append("age", f.age.value);
    fd.append("sex", f.sex.value);
    fd.append("exam_date", f.exam_date.value);
    if (f.left_image.files[0]) fd.append("left_image", f.left_image.files[0]);
    if (f.right_image.files[0]) fd.append("right_image", f.right_image.files[0]);
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Analysing…';
    try {
      const res = await api("/api/examinations", { method: "POST", body: fd });
      location.hash = `#/visit/${res.visit_id}`;
    } catch (e) {
      err.textContent = e.message; btn.disabled = false; btn.textContent = "Analyse images";
    }
  });
}

// ------------------------------------------------------------------ results
function probRows(e) {
  if (!e.disease_probs) return `<p class="muted small">No disease model is available.</p>`;
  const order = META.labels.map((l) => l.code);
  return order.map((code) => {
    const p = e.disease_probs[code], sd = (e.disease_uncertainty || {})[code] || 0, th = (e.disease_thresholds || {})[code];
    const lab = META.labels.find((l) => l.code === code);
    const above = code !== "N" && th != null && p >= th;
    const lo = Math.max(0, p - sd), hi = Math.min(1, p + sd);
    return `<div class="prob ${above ? "above" : ""} ${code === "N" ? "normal" : ""}" title="${esc(lab.name)}: ${pct(p, 1)} ± ${pct(sd, 1)} (model threshold ${pct(th)})">
      <div class="name">${esc(lab.short)}</div>
      <div class="track"><div class="fill" style="width:${100 * p}%"></div>
        <div class="band" style="left:${100 * lo}%;width:${100 * (hi - lo)}%"></div>
        ${th != null ? `<div class="tick" style="left:calc(${100 * th}% - 1px)"></div>` : ""}</div>
      <div class="val">${pct(p)}</div></div>`;
  }).join("") + `<div class="legend"><span><i class="lg-tick"></i>model threshold</span><span><i class="lg-band"></i>± uncertainty</span><span><i class="lg-flag"></i>score above threshold</span></div>`;
}

function seBlock(e) {
  if (e.predicted_se == null) return `<p class="muted small">No refractive-error model is available, so no spherical-equivalent estimate is shown.</p>`;
  const lo = -14, hi = 6, x = (v) => `${(100 * (Math.min(hi, Math.max(lo, v)) - lo)) / (hi - lo)}%`;
  const ticks = [-12, -9, -6, -3, 0, 3, 6].map((t) => `<span class="se-tick" style="left:${x(t)}">${t > 0 ? "+" + t : t}</span>`).join("");
  return `${e.refractive_is_demo ? `<div class="demo-box">Demo model trained on <b>synthetic images only</b>. This number is meaningless for real photographs — it only shows how the feature works.</div>` : ""}
    <div class="se-box">
      <div class="se-meta"><span>Estimated spherical equivalent</span><span>Model confidence ${pct(e.se_confidence)}</span></div>
      <div class="se-value">${dio(quarter(e.predicted_se))}</div>
      <div class="se-line" aria-hidden="true"><div class="se-axis"></div>
        <div class="se-int" style="left:${x(e.se_interval_low)};width:calc(${x(e.se_interval_high)} - ${x(e.se_interval_low)})"></div>
        <div class="se-mark" style="left:calc(${x(e.predicted_se)} - 1px)"></div>${ticks}</div>
      <div class="se-meta"><span>Approx. 95% range ${dio(e.se_interval_low)} to ${dio(e.se_interval_high)}</span></div>
      <div class="not-rx"><b>Experimental AI estimate. Not a prescription.</b> Professional refraction is required for an actual prescription.</div>
    </div>
    <p class="tiny muted" style="margin-top:6px">Confidence = estimated probability that the true value lies within ±1.00 D of the estimate.</p>`;
}

function qualityBlock(e) {
  const q = e.quality;
  if (!q || !q.warnings || !q.warnings.length) return "";
  return `<div class="warn-box"><b>${q.gradable ? "Image quality notes" : "Image may be unsuitable — results unreliable"}</b>
    <ul>${q.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`;
}

function eyeColumn(eye, e) {
  if (!e) return `<div class="eye-col"><div class="eye-title"><h2>${EYE[eye]}<span>${EYE_ABBR[eye]}</span></h2></div>
    <p class="muted">No image uploaded for this eye.</p></div>`;
  const flagged = e.disease_probs ? Object.entries(e.disease_probs)
    .filter(([c, p]) => c !== "N" && e.disease_thresholds && p >= e.disease_thresholds[c])
    .map(([c]) => META.labels.find((l) => l.code === c).name) : [];
  const summary = !e.disease_probs ? "" : flagged.length
    ? `Elevated model score for ${flagged.map(esc).join(", ")}-like features. This is not a diagnosis.`
    : "No class exceeded the model's threshold. This does not rule out eye disease.";
  return `<div class="eye-col">
    <div class="eye-title"><h2>${EYE[eye]}<span>${EYE_ABBR[eye]}</span></h2>
      ${e.disease_is_demo ? '<span class="tag demo">demo disease model</span>'
        : `<span class="tag real" title="${esc(e.disease_model_version || "")}">${/-odir/.test(e.disease_model_version || "") ? "ODIR-5K model" : "trained model"}</span>`}</div>
    ${e.disease_is_demo ? `<div class="demo-box">Disease scores come from a demo model trained on synthetic images.</div>` : ""}
    <div class="fundus-wrap"><img class="fundus" id="img-${eye}" src="${esc(e.image_url)}" alt="${EYE[eye]} fundus photograph"></div>
    ${e.heatmap_url ? `<div class="view-toggle" role="group" aria-label="Image view">
        <button class="on" data-eye="${eye}" data-src="${esc(e.image_url)}">Photo</button>
        <button data-eye="${eye}" data-src="${esc(e.heatmap_url)}">Model attention (${esc(e.heatmap_class || "")})</button></div>
      <div class="cam-note">Model attention visualization — research use only. It does not prove the model is looking at the correct pathology.</div>` : ""}
    ${qualityBlock(e)}
    <div class="section-label"><span>Disease pattern analysis</span><span>confidence ${pct(e.disease_confidence)}</span></div>
    ${probRows(e)}
    ${summary ? `<div class="summary-line">${summary}</div>` : ""}
    <div class="section-label"><span>Vision estimate</span></div>
    ${seBlock(e)}
  </div>`;
}

async function renderVisit(id) {
  const v = await api(`/api/visits/${encodeURIComponent(id)}`);
  const p = v.patient;
  view.innerHTML = `
    <div class="page-head"><div><h1>Results</h1>
      <p>Patient <a href="#/patient/${encodeURIComponent(p.patient_id)}">${esc(p.patient_id)}</a>
      · ${esc(v.exam_date)}${p.age != null ? ` · age ${esc(p.age)}` : ""}${p.sex && p.sex !== "U" ? ` · ${esc(p.sex)}` : ""}</p></div>
      <div style="display:flex;gap:10px"><a class="btn ghost" href="#/trends/${encodeURIComponent(p.patient_id)}">View trends</a>
      <a class="btn" href="#/new/${encodeURIComponent(p.patient_id)}">Add examination</a></div></div>
    ${disclaimer()}
    <div class="eye-cols">${eyeColumn("R", v.eyes.R)}${eyeColumn("L", v.eyes.L)}</div>
    <p class="tiny muted" style="margin-top:14px">Probabilities are the mean of 10 Monte-Carlo-dropout passes; the shaded band is ±1 standard deviation across passes.
    "Confidence" for disease scores measures how decisive the model is (1 − mean binary entropy), not its accuracy.
    See <a href="#/models">Model information</a> for tested performance.</p>`;
  view.querySelectorAll(".view-toggle button").forEach((b) => b.addEventListener("click", () => {
    b.parentElement.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    document.getElementById(`img-${b.dataset.eye}`).src = b.dataset.src;
  }));
}

// ------------------------------------------------------------------ history
async function renderPatients() {
  const ps = await api("/api/patients");
  view.innerHTML = `
    <div class="page-head"><div><h1>History</h1><p>Choose a patient to see all of their stored examinations.</p></div>
      <a class="btn" href="#/new">New examination</a></div>
    <div class="panel">${ps.length ? `<table class="data"><thead><tr><th>Patient</th><th>Age</th><th>Sex</th><th class="num">Visits</th><th>Last examination</th><th>Registered</th></tr></thead><tbody>
      ${ps.map((p) => `<tr class="click" onclick="location.hash='#/patient/${encodeURIComponent(p.patient_id)}'">
        <td><b>${esc(p.patient_id)}</b></td><td>${p.age ?? "–"}</td><td>${esc(p.sex || "U")}</td><td class="num">${p.n_visits}</td>
        <td>${esc(p.last_exam || "–")}</td><td>${esc(p.created_date)}</td></tr>`).join("")}</tbody></table>`
    : `<div class="empty">No patients yet.<br><a class="btn" href="#/new">New examination</a></div>`}</div>`;
}

async function renderPatient(pid) {
  const d = await api(`/api/patients/${encodeURIComponent(pid)}`);
  const top = (e) => {
    if (!e || !e.disease_probs) return "–";
    const [c, p] = Object.entries(e.disease_probs).filter(([c]) => c !== "N").sort((a, b) => b[1] - a[1])[0];
    const above = e.disease_thresholds && p >= e.disease_thresholds[c];
    return `<span class="${above ? "tag flag" : ""}">${esc(META.labels.find((l) => l.code === c).short)} ${pct(p)}</span>`;
  };
  view.innerHTML = `
    <div class="page-head"><div><h1>Patient ${esc(pid)}</h1>
      <p>${d.patient.age != null ? `Age ${esc(d.patient.age)} · ` : ""}${d.patient.sex !== "U" ? `${esc(d.patient.sex)} · ` : ""}registered ${esc(d.patient.created_date)}</p></div>
      <div style="display:flex;gap:10px"><a class="btn ghost" href="#/trends/${encodeURIComponent(pid)}">Trends</a>
      <a class="btn" href="#/new/${encodeURIComponent(pid)}">Add examination</a></div></div>
    ${disclaimer()}
    <div class="panel"><table class="data"><thead><tr><th>Date</th>
      <th>OD highest score</th><th class="num">OD est. SE</th><th>OS highest score</th><th class="num">OS est. SE</th><th></th></tr></thead><tbody>
      ${d.visits.map((v) => `<tr class="click" onclick="location.hash='#/visit/${esc(v.visit_id)}'">
        <td>${esc(v.exam_date)}</td>
        <td>${top(v.eyes.R)}</td><td class="num">${v.eyes.R && v.eyes.R.predicted_se != null ? dio(quarter(v.eyes.R.predicted_se)) : "–"}</td>
        <td>${top(v.eyes.L)}</td><td class="num">${v.eyes.L && v.eyes.L.predicted_se != null ? dio(quarter(v.eyes.L.predicted_se)) : "–"}</td>
        <td>${Object.values(v.eyes).some((e) => e.disease_is_demo || e.refractive_is_demo) ? '<span class="tag demo">demo</span>' : ""}</td></tr>`).join("")}
    </tbody></table></div>`;
}

// ------------------------------------------------------------------ trends
async function renderTrends(pid) {
  const ps = await api("/api/patients");
  if (!ps.length) {
    view.innerHTML = `<div class="page-head"><div><h1>Trends</h1></div></div><div class="panel empty">No patients yet.<br><a class="btn" href="#/new">New examination</a></div>`;
    return;
  }
  if (!pid) { location.hash = `#/trends/${encodeURIComponent(ps[0].patient_id)}`; return; }
  const t = await api(`/api/patients/${encodeURIComponent(pid)}/trends`);
  const eyes = ["R", "L"].filter((e) => t.eyes[e]);
  view.innerHTML = `
    <div class="page-head"><div><h1>Trends</h1><p>How the model's estimates for one patient changed between examinations.</p></div>
      <label class="field" style="min-width:220px">Patient<select id="pick">${ps.map((p) =>
        `<option value="${esc(p.patient_id)}" ${p.patient_id === pid ? "selected" : ""}>${esc(p.patient_id)} (${p.n_visits} visit${p.n_visits === 1 ? "" : "s"})</option>`).join("")}</select></label></div>
    ${disclaimer()}
    <div class="warn-box" style="margin:0 0 16px">${esc(t.note)}</div>
    ${eyes.map((e) => trendPanel(e, t.eyes[e])).join("")}`;
  document.getElementById("pick").addEventListener("change", (ev) => { location.hash = `#/trends/${encodeURIComponent(ev.target.value)}`; });
  eyes.forEach((e) => drawTrend(e, t.eyes[e]));
}

function trendPanel(eye, d) {
  const c = d.se_change;
  let change = `<p class="muted small">At least two examinations with a refractive estimate are needed to show a change.</p>`;
  if (c) {
    const sig = c.exceeds_uncertainty
      ? "The change is larger than the combined model uncertainty of the two estimates."
      : "The change is within the combined model uncertainty — it may be noise.";
    change = `<div class="change-card">
      <div><div class="small muted">Estimated change ${esc(c.from_date)} → ${esc(c.to_date)}</div><div class="delta">${dio(c.delta_se)}</div></div>
      <div><div class="small muted">Since first estimate (${esc(c.first_date)})</div><div class="delta" style="font-size:1.15rem">${dio(c.delta_since_first)}</div></div>
      <div class="small" style="max-width:420px">${sig}${c.same_model_version ? "" : " <b>The two estimates came from different model versions</b>, so the difference may reflect the model, not the eye."}</div></div>`;
  }
  const pc = d.prob_change ? Object.entries(d.prob_change).filter(([k]) => k !== "N")
    .map(([k, v]) => { const r = Math.round(100 * v); return `<span class="tag" style="margin:2px">${esc(META.labels.find((l) => l.code === k).short)} ${r > 0 ? "+" : r < 0 ? "−" : "±"}${Math.abs(r)} pts</span>`; }).join("") : "";
  const demo = d.series.some((s) => s.demo);
  return `<div class="panel">
    <div class="panel-head"><h2>${EYE[eye]} (${EYE_ABBR[eye]})</h2><span class="hint">${d.series.length} examination${d.series.length === 1 ? "" : "s"}</span></div>
    ${demo ? `<div class="demo-box">Some of these values come from demo models trained on synthetic images.</div>` : ""}
    ${change}
    <div class="grid-2" style="margin-top:14px">
      <div><h3 style="margin-bottom:6px">Estimated spherical equivalent</h3><div class="chart-box"><canvas id="se-${eye}"></canvas></div></div>
      <div><h3 style="margin-bottom:6px">Disease probabilities</h3><div class="chart-box"><canvas id="pr-${eye}"></canvas></div></div>
    </div>
    ${pc ? `<div class="small muted" style="margin-top:10px">Change in probability since the previous examination: ${pc}</div>` : ""}
  </div>`;
}

const PALETTE = { D: "#2a78d6", G: "#eb6834", C: "#1baf7a", A: "#eda100", H: "#e87ba4", M: "#008300", O: "#4a3aa7" };

function drawTrend(eye, d) {
  const labels = d.series.map((s) => s.exam_date);
  const se = document.getElementById(`se-${eye}`);
  if (d.series.some((s) => s.se != null)) {
    charts.push(new Chart(se, {
      type: "line",
      data: { labels, datasets: [
        { label: "95% range (high)", data: d.series.map((s) => s.se_high), borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: "rgba(192,84,27,.13)" },
        { label: "95% range (low)", data: d.series.map((s) => s.se_low), borderWidth: 0, pointRadius: 0, fill: false },
        { label: "Estimated SE (D)", data: d.series.map((s) => s.se), borderColor: "#c0541b", backgroundColor: "#c0541b", pointRadius: 4, tension: 0 },
      ] },
      options: { maintainAspectRatio: false, plugins: { legend: { labels: { filter: (i) => i.datasetIndex === 2 } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y?.toFixed(2)} D` } } },
        scales: { y: { title: { display: true, text: "diopters" } } } },
    }));
  } else { se.parentElement.innerHTML = `<p class="muted small">No refractive estimates.</p>`; }
  const pr = document.getElementById(`pr-${eye}`);
  charts.push(new Chart(pr, {
    type: "line",
    data: { labels, datasets: Object.keys(PALETTE).map((k) => ({
      label: META.labels.find((l) => l.code === k).short, data: d.series.map((s) => s.probs ? 100 * s.probs[k] : null),
      borderColor: PALETTE[k], backgroundColor: PALETTE[k], pointRadius: 3, tension: 0 })) },
    options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 100, title: { display: true, text: "probability (%)" } } },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } } },
  }));
}

// ------------------------------------------------------------------ model info
function metricTable(rows) {
  return `<table class="data"><tbody>${rows.map(([k, v]) => `<tr><td class="muted">${esc(k)}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table>`;
}
const f3 = (v) => (v == null ? "–" : Number(v).toFixed(3));

async function renderModels() {
  const m = await api("/api/model-info");
  const dis = m.disease, ref = m.refractive;
  const disBlock = !dis ? `<p class="muted">No disease model found. Train one with <code>python train.py --task disease</code>.</p>` : `
    ${dis.is_demo ? `<div class="demo-box">Demo model trained on synthetic images — its metrics only show that the pipeline runs.</div>` : ""}
    <dl class="kv">
      <dt>Model version</dt><dd>${esc(dis.model_version)}</dd>
      <dt>Architecture</dt><dd>${esc(dis.arch)} (ImageNet-pretrained: ${dis.pretrained_imagenet ? "yes" : "no"}), ${esc(dis.img_size)}×${esc(dis.img_size)} px input</dd>
      <dt>Dataset</dt><dd>${esc(dis.dataset)} — ${esc(dis.n_train_images)} training images from ${esc(dis.n_train_patients)} patients (patient-level split)</dd>
      <dt>Loss</dt><dd>${esc(dis.loss)}</dd>
      <dt>Training date</dt><dd>${esc(dis.train_date)}</dd>
      <dt>Thresholds</dt><dd>${esc(dis.threshold_strategy || "")}</dd>
    </dl>
    <div class="grid-2" style="margin-top:14px">
      <div><h3>Validation</h3>${metricTable([["Macro ROC-AUC", f3(dis.val_metrics?.macro_roc_auc)], ["Macro PR-AUC", f3(dis.val_metrics?.macro_pr_auc)], ["Macro F1", f3(dis.val_metrics?.macro_f1)]])}</div>
      <div><h3>Test (held-out patients)</h3>${dis.test_metrics ? metricTable([["Macro ROC-AUC", f3(dis.test_metrics.macro_roc_auc)], ["Macro PR-AUC", f3(dis.test_metrics.macro_pr_auc)],
        ["Macro F1", f3(dis.test_metrics.macro_f1)], ["Macro sensitivity", f3(dis.test_metrics.macro_recall)], ["Macro specificity", f3(dis.test_metrics.macro_specificity)], ["Micro F1", f3(dis.test_metrics.micro_f1)]])
        : `<p class="muted small">Not evaluated yet — run <code>python evaluate.py --task disease</code>.</p>`}</div>
    </div>
    ${dis.report_pdf ? `<p style="margin-top:12px"><a class="btn ghost" href="${esc(dis.report_pdf)}" target="_blank" rel="noopener">Open evaluation report (PDF)</a></p>` : ""}
    <div class="plots">${(dis.plots || []).map((p) => `<img src="${esc(p)}" alt="evaluation plot" loading="lazy">`).join("")}
      ${dis.history_plot ? `<img src="${esc(dis.history_plot)}" alt="training history" loading="lazy">` : ""}</div>`;
  const refBlock = !ref ? `<p class="muted">No refractive-error model. It needs fundus photos paired with measured refraction (see README).</p>` : `
    ${ref.is_demo ? `<div class="demo-box">Demo model trained on <b>synthetic</b> images with synthetic refraction labels. No public fundus + refraction dataset could be downloaded, so this model has no clinical meaning.</div>` : ""}
    <dl class="kv"><dt>Model version</dt><dd>${esc(ref.model_version)}</dd><dt>Architecture</dt><dd>${esc(ref.arch)} regression head</dd>
      <dt>Dataset</dt><dd>${esc(ref.dataset)} (${esc(ref.n_train_images)} training images)</dd><dt>Training date</dt><dd>${esc(ref.train_date)}</dd>
      <dt>Validation residual SD</dt><dd>${ref.residual_std != null ? Number(ref.residual_std).toFixed(2) + " D" : "–"}</dd></dl>
    <div class="grid-2" style="margin-top:14px">
      <div><h3>Validation</h3>${metricTable([["MAE (D)", f3(ref.val_metrics?.mae)], ["RMSE (D)", f3(ref.val_metrics?.rmse)], ["R²", f3(ref.val_metrics?.r2)], ["Within ±0.50 D", pct(ref.val_metrics?.["within_0.50D"])], ["Within ±1.00 D", pct(ref.val_metrics?.["within_1.00D"])]])}</div>
      <div><h3>Test</h3>${ref.test_metrics ? metricTable([["MAE (D)", f3(ref.test_metrics.mae)], ["RMSE (D)", f3(ref.test_metrics.rmse)], ["R²", f3(ref.test_metrics.r2)], ["Within ±0.50 D", pct(ref.test_metrics["within_0.50D"])], ["Within ±1.00 D", pct(ref.test_metrics["within_1.00D"])], ["Within ±2.00 D", pct(ref.test_metrics["within_2.00D"])]]) : `<p class="muted small">Not evaluated.</p>`}</div>
    </div>
    <div class="plots">${(ref.plots || []).map((p) => `<img src="${esc(p)}" alt="evaluation plot" loading="lazy">`).join("")}</div>`;
  const comp = m.comparison && m.comparison.length ? `<div class="panel"><div class="panel-head"><h2>Model comparison (test set)</h2></div>
    <table class="data"><thead><tr><th>Model</th><th class="num">Macro ROC-AUC</th><th class="num">Macro PR-AUC</th><th class="num">Macro F1</th></tr></thead><tbody>
    ${m.comparison.map((r) => `<tr><td>${esc(r.model)}</td><td class="num">${f3(r.macro_roc_auc)}</td><td class="num">${f3(r.macro_pr_auc)}</td><td class="num">${f3(r.macro_f1)}</td></tr>`).join("")}
    </tbody></table></div>` : "";
  view.innerHTML = `
    <div class="page-head"><div><h1>Model information</h1><p>What the models were trained on and how they performed on held-out data.</p></div></div>
    ${disclaimer()}
    <div class="panel"><div class="panel-head"><h2>Disease classifier</h2>${dis ? (dis.is_demo ? '<span class="tag demo">demo</span>' : '<span class="tag real">trained on real data</span>') : ""}</div>${disBlock}</div>
    ${comp}
    <div class="panel"><div class="panel-head"><h2>Refractive-error estimator</h2>${ref ? (ref.is_demo ? '<span class="tag demo">demo</span>' : '<span class="tag real">trained on real data</span>') : ""}</div>${refBlock}</div>
    <div class="panel"><h2 style="margin-bottom:8px">How to read the numbers</h2>
      <p class="small">ROC-AUC measures how well the model ranks positive above negative images (0.5 = chance). PR-AUC should be compared with the class prevalence,
      which is its value for a random model. Sensitivity and specificity depend on the decision threshold, which was tuned on the validation set only.
      Small classes give unstable estimates. None of these figures show clinical validity.</p></div>`;
}
