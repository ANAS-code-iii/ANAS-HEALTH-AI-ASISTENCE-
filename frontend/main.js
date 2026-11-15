/* main.js - all browser logic. Talks only to our own /api/* endpoints. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);

// ---- shared state --------------------------------------------------------
const state = {
  history: [],        // chat turns sent back to the server for context
  diseases: [],       // specs from /api/diseases
  current: null,      // currently selected disease spec
  predictions: {},    // latest ML result per disease id (reused by the agent)
};

// ---- helpers -------------------------------------------------------------
async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = "Please check your input.";   // FastAPI validation error
    throw new Error(msg || `Server error (${res.status})`);
  }
  return data;
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  children.flat().forEach(c => c != null && node.append(c));
  return node;
}

/** Render markdown safely (falls back to plain text if CDN libs are blocked). */
function renderMarkdown(target, text) {
  target.classList.add("md");
  if (window.marked && window.DOMPurify) {
    target.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } else {
    target.textContent = text;
    target.style.whiteSpace = "pre-wrap";
  }
}

function emergencyBanner(flags) {
  if (!flags || !flags.length) return null;
  return el("div", { class: "alert", role: "alert" },
    `⚠️ Possible emergency signs mentioned (${flags.join(", ")}). Call 999 or go to the nearest hospital now – do not wait for this app.`);
}

function setBusy(btn, busy, idle) {
  btn.disabled = busy;
  btn.textContent = busy ? "Please wait…" : idle;
}

// ---- tabs ----------------------------------------------------------------
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => {
      const on = t === tab;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on);
    });
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    $("#tab-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "report") refreshAttachList();
  });
});

/* =========================================================================
   1. CHAT
   ========================================================================= */
const messagesEl = $("#messages");
const chatInput = $("#chat-input");

let pendingFiles = [];

function renderFiles() {
  const box = $("#file-list");
  box.hidden = pendingFiles.length === 0;
  box.replaceChildren(...pendingFiles.map((f, i) =>
    el("span", { class: "file-chip" }, `📄 ${f.name} `,
      el("button", { type: "button", "aria-label": "Remove file", onclick: () => { pendingFiles.splice(i, 1); renderFiles(); } }, "×"))));
}

function mlCard(ml) {
  if (!ml) return null;
  if (ml.status === "done") {
    const yes = ml.verdict === "YES";
    return el("div", { class: "ml-card " + (yes ? "yes" : "no") },
      el("div", { class: "verdict" },
        el("span", { class: "badge", text: ml.verdict }),
        el("div", {}, el("strong", { text: `${ml.name}: ${ml.result_label}` }),
          el("div", { class: "muted", text: `Model confidence ${ml.confidence_percent}%` }))),
      el("p", { class: "next", text: yes
        ? "ML screening signal only – not a diagnosis. Please see a doctor for proper tests."
        : "No risk flagged by this model, but this does not rule out disease." }));
  }
  const need = ml.missing && ml.missing.length ? ` I still need: ${ml.missing.join(", ")}.` : (ml.error ? " " + ml.error : "");
  return el("div", { class: "ml-card need" },
    `To run the ${ml.name} YES/NO screening.${need} Open the “Disease check” tab to enter the values.`);
}

function addMessage(role, text, flags, ml, fileNames) {
  const bubble = el("div", { class: "bubble" });
  if (role === "bot") renderMarkdown(bubble, text); else bubble.textContent = text;
  if (fileNames && fileNames.length) bubble.append(el("div", { class: "muted", text: "📎 " + fileNames.join(", ") }));
  const wrap = el("div", { class: `msg ${role}` }, bubble);
  const card = mlCard(ml);
  if (card) bubble.append(card);
  const banner = emergencyBanner(flags);
  if (banner) bubble.prepend(banner);
  messagesEl.append(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

async function sendChat(text) {
  text = text.trim();
  const files = pendingFiles.slice();
  if (!text && !files.length) return;
  $("#chips").hidden = true;
  addMessage("user", text || "(attached file)", null, null, files.map(f => f.name));
  chatInput.value = "";
  pendingFiles = [];
  renderFiles();
  autoGrow();

  const typing = el("div", { class: "msg bot" },
    el("div", { class: "bubble typing", "aria-label": "AI is typing" },
      el("span"), el("span"), el("span")));
  messagesEl.append(typing);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  setBusy($("#chat-send"), true, "Send");

  try {
    const fd = new FormData();
    fd.append("message", text);
    fd.append("history", JSON.stringify(state.history));
    files.forEach(f => fd.append("files", f));
    const res = await fetch("/api/ask", { method: "POST", body: fd });
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) {
      let msg = data && data.detail;
      if (Array.isArray(msg)) msg = "Please check your input.";
      throw new Error(msg || `Server error (${res.status})`);
    }
    typing.remove();
    addMessage("bot", data.answer, data.emergency_flags, data.ml);
    state.history.push({ role: "user", content: text || "(attached file)" }, { role: "assistant", content: data.answer });
    state.history = state.history.slice(-12);
  } catch (err) {
    typing.remove();
    addMessage("bot", "Sorry, something went wrong: " + err.message);
  } finally {
    setBusy($("#chat-send"), false, "Send");
    chatInput.focus();
  }
}

$("#attach-btn").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", e => {
  pendingFiles = pendingFiles.concat([...e.target.files]).slice(0, 4);
  e.target.value = "";
  renderFiles();
});

function autoGrow() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
}

$("#chat-form").addEventListener("submit", e => { e.preventDefault(); sendChat(chatInput.value); });
chatInput.addEventListener("input", autoGrow);
chatInput.addEventListener("keydown", e => {           // Enter = send, Shift+Enter = new line
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
document.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => sendChat(c.textContent)));

/* =========================================================================
   2. DISEASE CHECK (ML)
   ========================================================================= */
async function loadDiseases() {
  const list = $("#disease-list");
  try {
    state.diseases = await api("/api/diseases");
  } catch (err) {
    list.replaceChildren(el("p", { class: "error", text: "Could not load disease models: " + err.message }));
    return;
  }
  renderDiseaseCards();
}

function renderDiseaseCards() {
  const list = $("#disease-list");
  list.replaceChildren(...state.diseases.map(d => {
    const done = state.predictions[d.id];
    const card = el("button", {
      class: "d-card" + (state.current && state.current.id === d.id ? " selected" : ""),
      type: "button",
      onclick: () => selectDisease(d),
    },
      el("strong", { text: d.name }),
      el("small", { text: d.ready ? d.description : "Model not installed yet" }),
      done ? el("small", { class: "done", text: ` ✓ Result: ${done.verdict}` }) : null);
    if (!d.ready) card.disabled = true;
    return card;
  }));
}

function selectDisease(d) {
  state.current = d;
  renderDiseaseCards();
  $("#form-card").hidden = false;
  $("#form-title").textContent = d.name;
  $("#result").hidden = true;
  $("#predict-error").hidden = true;

  const fields = $("#fields");
  fields.replaceChildren(...d.fields.map(f => {
    const id = "f_" + f.key.replace(/\W+/g, "_");
    let input;
    if (f.type === "select") {
      input = el("select", { id, name: f.key, required: "" },
        el("option", { value: "", text: "Select…", disabled: "", selected: "" }),
        ...f.options.map(o => el("option", { value: o, text: o })));
    } else {
      input = el("input", {
        id, name: f.key, type: "number", inputmode: "decimal", step: "any",
        min: f.min, max: f.max, required: "", placeholder: `${f.min} – ${f.max}`,
      });
    }
    const label = el("label", { for: id }, f.label,
      f.unit ? el("span", { class: "unit", text: ` (${f.unit})` }) : null);
    return el("div", {}, label, input);
  }));
  $("#form-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("#predict-form").addEventListener("submit", async e => {
  e.preventDefault();
  const d = state.current;
  if (!d) return;
  const errBox = $("#predict-error");
  errBox.hidden = true;

  // collect + basic client-side validation (server validates again)
  const inputs = {};
  let firstBad = null;
  d.fields.forEach(f => {
    const input = $(`[name="${CSS.escape(f.key)}"]`, $("#fields"));
    const val = input.value.trim();
    const bad = val === "" || (f.type === "number" && (isNaN(val) || +val < f.min || +val > f.max));
    input.classList.toggle("invalid", bad);
    if (bad && !firstBad) firstBad = { input, f };
    inputs[f.key] = f.type === "number" ? Number(val) : val;
  });
  if (firstBad) {
    errBox.textContent = firstBad.f.type === "number"
      ? `${firstBad.f.label}: enter a number between ${firstBad.f.min} and ${firstBad.f.max}.`
      : `Please choose: ${firstBad.f.label}.`;
    errBox.hidden = false;
    firstBad.input.focus();
    return;
  }

  setBusy($("#predict-btn"), true, "Check now");
  try {
    const r = await api("/api/predict", { disease: d.id, inputs });
    state.predictions[d.id] = r;
    showResult(r);
    renderDiseaseCards();
  } catch (err) {
    errBox.textContent = err.message;
    errBox.hidden = false;
  } finally {
    setBusy($("#predict-btn"), false, "Check now");
  }
});

function showResult(r) {
  const box = $("#result");
  const yes = r.verdict === "YES";
  box.className = "result " + (yes ? "yes" : "no");
  box.hidden = false;
  box.replaceChildren(
    el("div", { class: "verdict" },
      el("span", { class: "badge", text: r.verdict }),
      el("div", {},
        el("strong", { text: r.result_label }),
        el("div", { class: "muted", text: `Model confidence ${r.confidence_percent}%` }))),
    el("div", { class: "bars" },
      ...Object.entries(r.probabilities).map(([label, pct]) =>
        el("div", { class: "bar-row" },
          el("span", { text: label }),
          el("div", { class: "bar" }, el("i", { style: `width:${pct}%` })),
          el("span", { text: pct + "%" })))),
    el("p", { class: "next" },
      yes ? "This suggests you may be at risk. Please see a doctor for proper tests. "
          : "No risk flagged by this model, but this does not rule out disease. ",
      "Use the “Health report” tab to get a full AI analysis that includes this result."));
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* =========================================================================
   3. AI AGENT (health report)
   ========================================================================= */
function refreshAttachList() {
  const items = Object.values(state.predictions);
  const box = $("#attach-box");
  box.hidden = items.length === 0;
  $("#attach-list").replaceChildren(...items.map(p =>
    el("label", {},
      el("input", { type: "checkbox", checked: "", value: p.disease }),
      `${p.name}: ${p.verdict} (${p.result_label})`)));
}

$("#agent-form").addEventListener("submit", async e => {
  e.preventDefault();
  const err = $("#agent-error");
  err.hidden = true;
  const symptoms = $("#symptoms").value.trim();
  if (!symptoms) {
    err.textContent = "Please describe your symptoms or medical history first.";
    err.hidden = false;
    return;
  }
  const chosen = [...document.querySelectorAll("#attach-list input:checked")].map(i => state.predictions[i.value]);

  setBusy($("#agent-btn"), true, "Generate report");
  const card = $("#report-card");
  card.hidden = false;
  card.replaceChildren(el("p", { class: "muted", text: "The AI agent is analysing… this can take 10–30 seconds." }));
  try {
    const data = await api("/api/agent", {
      history_and_symptoms: symptoms,
      additional_information: $("#extra").value.trim(),
      predictions: chosen.map(({ disease, name, verdict, result_label, confidence_percent }) =>
        ({ disease, name, verdict, result_label, confidence_percent })),
    });
    const body = el("div");
    renderMarkdown(body, data.analysis);
    const banner = emergencyBanner(data.emergency_flags);
    card.replaceChildren(...(banner ? [banner] : []), body);
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (ex) {
    card.hidden = true;
    err.textContent = ex.message;
    err.hidden = false;
  } finally {
    setBusy($("#agent-btn"), false, "Generate report");
  }
});

loadDiseases();