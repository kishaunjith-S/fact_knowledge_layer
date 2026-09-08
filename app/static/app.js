const state = { verdictFilter: "" };

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url} -> ${response.status}`);
  return response.json();
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      loadTab(btn.dataset.tab);
    });
  });
}

function loadTab(tab) {
  if (tab === "documents") loadDocuments();
  if (tab === "facts") loadFacts();
  if (tab === "findings") loadFindings();
  if (tab === "failures") loadFailures();
}

async function loadDocuments() {
  const docs = await fetchJSON("/api/documents");
  const tbody = document.querySelector("#documents-table tbody");
  tbody.innerHTML = "";
  for (const doc of docs) {
    tbody.appendChild(el("tr", {}, [
      el("td", { text: doc.filename }),
      el("td", { text: doc.page_count }),
      el("td", { text: doc.claim_count ?? "-" }),
      el("td", { text: doc.doc_date || "-" }),
      el("td", { text: doc.status }),
    ]));
  }
}

async function loadFacts(query = "") {
  const url = query ? `/api/claims?q=${encodeURIComponent(query)}` : "/api/claims";
  const claims = await fetchJSON(url);
  const tbody = document.querySelector("#facts-table tbody");
  tbody.innerHTML = "";
  for (const claim of claims) {
    const value = claim.value_type === "numeric" ? claim.value_num : claim.value_text;
    tbody.appendChild(el("tr", {}, [
      el("td", { text: claim.subject }),
      el("td", { text: claim.measure }),
      el("td", { text: `${value} ${claim.unit_raw || ""}`.trim() }),
      el("td", { text: claim.period_label || "-" }),
      el("td", { text: `p.${claim.evidence_page}` }),
    ]));
  }
}

const VERDICT_LABEL = {
  CORROBORATES: "Corroborated",
  CONTRADICTS: "Contradicted",
  RECONCILED_BY_CONTEXT: "Reconciled by context",
  INSUFFICIENT_CONTEXT: "Needs review",
};

async function loadFindings() {
  const relations = await fetchJSON("/api/relations");
  const container = document.getElementById("findings-list");
  const shown = relations.filter((r) =>
    state.verdictFilter ? r.final_verdict === state.verdictFilter
                        : r.final_verdict !== "INSUFFICIENT_CONTEXT");
  container.innerHTML = "";
  if (!shown.length) {
    container.appendChild(el("p", { class: "empty-note", text: "No findings for this filter." }));
    return;
  }
  const enriched = await Promise.all(shown.map(async (r) => {
    const [a, b] = await Promise.all([
      fetchJSON(`/api/evidence/${r.claim_a_id}`).catch(() => null),
      fetchJSON(`/api/evidence/${r.claim_b_id}`).catch(() => null),
    ]);
    return { r, a, b };
  }));
  enriched.forEach(({ r, a, b }, i) => container.appendChild(renderFindingCard(r, a, b, i === 0)));
}

function envelopeRow(label, va, vb, differs) {
  return el("tr", { class: differs ? "env-differs" : "" }, [
    el("th", { text: label }),
    el("td", { text: va ?? "—" }),
    el("td", { text: vb ?? "—" }),
  ]);
}

function renderFindingCard(relation, a, b, startExpanded) {
  const card = el("div", { class: "finding-card" + (startExpanded ? " expanded" : "") });
  const measure = (a && a.measure) || (b && b.measure) || "";
  const header = el("div", { class: "finding-header" }, [
    el("div", { class: "finding-title" }, [
      el("span", { class: `verdict-badge verdict-${relation.final_verdict}`,
                   text: VERDICT_LABEL[relation.final_verdict] || relation.final_verdict }),
      el("span", { class: "finding-measure", text: measure }),
    ]),
    el("span", { class: "finding-axis", text: relation.axis ? `axis: ${relation.axis}` : "" }),
  ]);
  header.addEventListener("click", () => card.classList.toggle("expanded"));

  const body = el("div", { class: "finding-body" });

  const fmt = (c) => c && c.value != null ? `${c.value}${c.unit ? " " + c.unit : ""}` : "—";
  const table = el("table", { class: "envelope-table" }, [
    el("thead", {}, [el("tr", {}, [
      el("th", { text: "" }),
      el("th", { text: a && a.document ? a.document : "Document A" }),
      el("th", { text: b && b.document ? b.document : "Document B" }),
    ])]),
    el("tbody", {}, [
      envelopeRow("value", fmt(a), fmt(b), relation.final_verdict === "CONTRADICTS"),
      envelopeRow("period", a && a.period, b && b.period, relation.axis === "period"),
      envelopeRow("vintage", a && a.vintage, b && b.vintage, relation.axis === "vintage"),
    ]),
  ]);
  body.appendChild(table);

  body.appendChild(el("div", { class: "evidence-pair" }, [
    el("div", { class: "evidence-quote", text: a ? `p.${a.page}  "${a.quote}"` : "(evidence unavailable)" }),
    el("div", { class: "evidence-quote", text: b ? `p.${b.page}  "${b.quote}"` : "(evidence unavailable)" }),
  ]));

  body.appendChild(el("div", { class: "rule-line", text: `rule fired:  ${relation.rule_fired}` }));
  if (relation.delta_pct != null) {
    body.appendChild(el("div", { class: "rule-line",
      text: `value gap:  ${(relation.delta_pct * 100).toFixed(1)}%` }));
  }
  if (relation.explanation) body.appendChild(el("div", { class: "explanation", text: relation.explanation }));
  if (relation.llm_overrode) {
    body.appendChild(el("div", { class: "override-note",
      text: `LLM overrode the rule (rule said ${relation.rule_verdict}): ${relation.override_reason || ""}` }));
  }
  card.appendChild(header);
  card.appendChild(body);
  return card;
}

async function loadFailures() {
  const failures = await fetchJSON("/api/failures");
  const tbody = document.querySelector("#failures-table tbody");
  tbody.innerHTML = "";
  for (const failure of failures) {
    tbody.appendChild(el("tr", {}, [
      el("td", { text: failure.reason }),
      el("td", { text: failure.kind }),
      el("td", { text: JSON.stringify(failure.raw_payload || {}).slice(0, 120) }),
    ]));
  }
}

function setupUpload() {
  document.getElementById("upload-btn").addEventListener("click", async () => {
    const input = document.getElementById("file-input");
    const status = document.getElementById("upload-status");
    if (!input.files.length) return;
    const formData = new FormData();
    formData.append("file", input.files[0]);
    status.textContent = "Uploading...";
    const response = await fetch("/api/documents", { method: "POST", body: formData });
    if (!response.ok) {
      status.textContent = "Upload failed.";
      return;
    }
    const body = await response.json();
    status.textContent = `Queued (doc_id: ${body.doc_id}). Ingesting in background...`;
    loadDocuments();
    const poll = setInterval(async () => {
      const doc = await fetchJSON(`/api/documents/${body.doc_id}`);
      loadDocuments();
      if (doc.status === "ready" || doc.status === "failed") {
        status.textContent = `Status: ${doc.status}`;
        clearInterval(poll);
      }
    }, 2000);
  });
}

function setupFindingFilters() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.verdictFilter = chip.dataset.verdict;
      loadFindings();
    });
  });
}

function setupSearch() {
  const input = document.getElementById("facts-search");
  let timeout;
  input.addEventListener("input", () => {
    clearTimeout(timeout);
    timeout = setTimeout(() => loadFacts(input.value), 300);
  });
}

setupTabs();
setupUpload();
setupFindingFilters();
setupSearch();
loadDocuments();
