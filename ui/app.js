"use strict";

const byId = (id) => document.getElementById(id);
const sessionId = window.crypto?.randomUUID?.() || `demo-${Date.now()}`;
let nursePassword = "";
let patients = [];

function addTextElement(parent, tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  parent.append(element);
  return element;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (nursePassword) headers.set("Authorization", `Basic ${btoa(`nurse:${nursePassword}`)}`);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function refreshMetrics() {
  try {
    const data = await api("/api/metrics");
    byId("metric-model").textContent = data.model || "Not configured";
    byId("metric-tps").textContent = Number(data.tokens_per_second || 0).toFixed(1);
    byId("metric-latency").textContent = `${Number(data.avg_latency_ms || 0).toFixed(0)} ms`;
    byId("metric-answered").textContent = String(data.counts?.answered || 0);
    byId("metric-escalated").textContent = String(
      (data.counts?.escalated || 0) + (data.counts?.red_flag || 0),
    );
  } catch (error) {
    if (byId("metric-model")) byId("metric-model").textContent = "Unavailable";
    console.error("Could not refresh service metrics.", error);
  }
}

/**
 * Fetch /api/about and apply run-mode UI state.
 * - demo mode: show red DEMO MODE banner on every page.
 * - live mode: hide banner; show AMD label only when amd_label_active is true.
 */
async function applyRunMode() {
  try {
    const about = await api("/api/about");
    const banner = byId("run-mode-banner");
    if (banner) {
      if (about.run_mode === "demo") {
        banner.removeAttribute("hidden");
        banner.setAttribute("data-run-mode", "demo");
      } else {
        banner.setAttribute("hidden", "");
        banner.setAttribute("data-run-mode", "live");
      }
    }
    const amdEl = byId("amd-status");
    if (amdEl) {
      amdEl.textContent = about.amd_label_active ? "Yes (AMD GPU)" : "No";
    }
    const amdLabel = document.querySelector(".amd-label");
    if (amdLabel) {
      amdLabel.hidden = !about.amd_label_active;
    }
  } catch (error) {
    // If /api/about fails, show the banner conservatively in demo style.
    const banner = byId("run-mode-banner");
    if (banner) banner.removeAttribute("hidden");
    console.warn("Could not fetch /api/about for run-mode banner.", error);
  }
}



function renderPatientMessage(message, result) {
  const conversation = byId("conversation");
  conversation.querySelector(".empty-state")?.remove();
  const userBubble = document.createElement("div");
  userBubble.className = "message user-message";
  addTextElement(userBubble, "p", message);
  conversation.append(userBubble);
  const assistantBubble = document.createElement("section");
  const statusClass = result.status === "answered" ? "badge-green" :
    result.status === "red_flag" ? "badge-red" : "badge-amber";
  assistantBubble.className = `message assistant-message ${statusClass}`;
  addTextElement(assistantBubble, "span", result.status.replace("_", " "), `status-badge ${statusClass}`);
  addTextElement(assistantBubble, "p", result.answer || "A nurse will follow up.");
  if (result.ticket_id) {
    addTextElement(assistantBubble, "p", `Nurse ticket: ${result.ticket_id}`, "ticket-reference");
  }
  if (result.language) byId("language-indicator").textContent = `Language: ${result.language}`;
  if (result.status === "answered") {
    addTextElement(assistantBubble, "p", `Confidence: ${(result.confidence * 100).toFixed(0)}%`, "confidence");
    const details = document.createElement("details");
    const summary = addTextElement(details, "summary", "Why this answer");
    summary.tabIndex = 0;
    for (const citation of result.citations || []) {
      addTextElement(details, "h3", citation.section);
      addTextElement(details, "p", citation.source_text, "citation-source");
      if (citation.translated_snippet) {
        addTextElement(details, "p", citation.translated_snippet, "citation-translation");
      }
    }
    const signals = Object.entries(result.breakdown || {});
    if (signals.length) {
      const list = document.createElement("ul");
      list.className = "signal-list";
      for (const [name, value] of signals) {
        addTextElement(list, "li", `${name.replaceAll("_", " ")}: ${Number(value).toFixed(2)}`);
      }
      details.append(list);
    }
    assistantBubble.append(details);
  }
  conversation.append(assistantBubble);
  conversation.scrollTop = conversation.scrollHeight;
}

async function loadPatients() {
  const select = byId("patient-select");
  if (!select) return;
  try {
    patients = await api("/api/patients");
    select.replaceChildren(new Option("Select a synthetic patient", ""));
    for (const patient of patients) {
      select.add(new Option(patient.patient_alias, patient.patient_id));
    }
  } catch (error) {
    select.replaceChildren(new Option("Patient list unavailable", ""));
    console.error("Could not load demo patients.", error);
  }
  select.addEventListener("change", () => {
    const patient = patients.find((item) => item.patient_id === select.value);
    byId("patient-document").textContent = patient ? patient.document_name : "";
  });
}

function setupChat() {
  const form = byId("chat-form");
  if (!form) return;
  const sendMessage = async (message) => {
    const patientId = byId("patient-select").value;
    if (!patientId) {
      byId("form-hint").textContent = "Choose a demo patient before sending a question.";
      return;
    }
    byId("send-button").disabled = true;
    byId("form-hint").textContent = "Checking your question safely…";
    try {
      const result = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ patient_id: patientId, message, session_id: sessionId }),
      });
      renderPatientMessage(message, result);
      byId("form-hint").textContent = "Do not enter real personal or medical information.";
      byId("message").value = "";
      await refreshMetrics();
    } catch (error) {
      byId("form-hint").textContent = error.message;
    } finally {
      byId("send-button").disabled = false;
    }
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = byId("message").value.trim();
    if (message) void sendMessage(message);
  });
  document.querySelectorAll(".chip").forEach((button) => {
    button.addEventListener("click", () => {
      byId("message").value = button.dataset.question || "";
      byId("message").focus();
    });
  });
}

function showCitation(parent, citation) {
  addTextElement(parent, "h4", `${citation.section} (${citation.chunk_id})`);
  addTextElement(parent, "p", citation.source_text, "citation-source");
}

function renderTicketDetail(ticket) {
  const detail = byId("ticket-detail");
  detail.replaceChildren();
  addTextElement(detail, "span", ticket.priority.toUpperCase(), `status-badge ${ticket.priority === "urgent" ? "badge-red" : "badge-amber"}`);
  addTextElement(detail, "h3", `Ticket ${ticket.id}`);
  addTextElement(detail, "p", `Patient: ${ticket.patient_id} · Language: ${ticket.language}`);
  addTextElement(detail, "h4", "Original message");
  addTextElement(detail, "p", ticket.original_message);
  addTextElement(detail, "h4", "English version");
  addTextElement(detail, "p", ticket.english_message || "Not available; translation failed.");
  addTextElement(detail, "h4", "Why it was escalated");
  addTextElement(detail, "p", ticket.reason);
  addTextElement(detail, "h4", "Confidence breakdown");
  const signals = document.createElement("ul");
  for (const [name, value] of Object.entries(ticket.breakdown || {})) {
    addTextElement(signals, "li", `${name}: ${Number(value).toFixed(2)}`);
  }
  detail.append(signals);
  for (const chunk of ticket.retrieved_chunks || []) showCitation(detail, chunk);
  if (ticket.draft_answer) {
    addTextElement(detail, "h4", "Unshown draft (review only)");
    addTextElement(detail, "p", ticket.draft_answer);
  }
  if (ticket.status === "open") {
    const form = document.createElement("form");
    form.className = "resolve-form";
    const label = addTextElement(form, "label", "Nurse reply");
    label.htmlFor = "nurse-reply";
    const textarea = document.createElement("textarea");
    textarea.id = "nurse-reply";
    textarea.maxLength = 4000;
    textarea.required = true;
    textarea.rows = 3;
    form.append(textarea);
    const button = document.createElement("button");
    button.className = "button-primary";
    button.type = "submit";
    button.textContent = "Resolve ticket";
    form.append(button);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        await api(`/api/tickets/${encodeURIComponent(ticket.id)}/resolve`, {
          method: "POST",
          body: JSON.stringify({ nurse_reply: textarea.value }),
        });
        await refreshTickets();
        detail.replaceChildren();
        addTextElement(detail, "p", "Ticket resolved.");
      } catch (error) {
        addTextElement(detail, "p", error.message, "error-message");
      }
    });
    detail.append(form);
  } else if (ticket.nurse_reply) {
    addTextElement(detail, "h4", "Nurse reply");
    addTextElement(detail, "p", ticket.nurse_reply);
  }
}

async function refreshTickets() {
  const list = byId("ticket-list");
  if (!list) return;
  try {
    const status = byId("ticket-filter").value;
    const tickets = await api(`/api/tickets?status=${status}`);
    list.replaceChildren();
    if (!tickets.length) addTextElement(list, "li", `No ${status} tickets.`, "empty-state");
    for (const ticket of tickets) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.className = "ticket-button";
      button.type = "button";
      addTextElement(button, "span", ticket.priority, `priority ${ticket.priority}`);
      addTextElement(button, "strong", ticket.original_message);
      addTextElement(button, "small", `${ticket.patient_id} · ${new Date(ticket.created_at).toLocaleString()}`);
      button.addEventListener("click", async () => {
        try {
          renderTicketDetail(await api(`/api/tickets/${encodeURIComponent(ticket.id)}`));
        } catch (error) {
          byId("queue-status").textContent = error.message;
        }
      });
      item.append(button);
      list.append(item);
    }
    byId("queue-status").textContent = `Showing ${tickets.length} ${status} ticket(s). Refreshes every 5 seconds.`;
  } catch (error) {
    byId("queue-status").textContent = `${error.message} Enter the nurse password if one is configured.`;
  }
}

function setupNurse() {
  if (!byId("ticket-list")) return;
  byId("save-password").addEventListener("click", () => {
    nursePassword = byId("nurse-password").value;
    void refreshTickets();
  });
  byId("ticket-filter").addEventListener("change", () => void refreshTickets());
  void refreshTickets();
  window.setInterval(() => void refreshTickets(), 5000);
}

void applyRunMode();
void loadPatients();
setupChat();
setupNurse();
void refreshMetrics();
window.setInterval(() => void refreshMetrics(), 10000);
