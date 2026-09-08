# Demo Video Script — Fact Knowledge Layer

**Target length: 3 minutes or less.** Record at 1280×720 or higher. Keep the
browser dev console closed. Have the app already running before you hit record:

```bash
python -m pip install -r requirements.txt
python scripts/seed_starter_corpus.py      # populates data/facts.db from the committed data/cache/ (no API key needed)
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`. Do a dry run once before recording so the tab
switches and the one upload are smooth.

---

## 0:00–0:20 — What this is (Documents tab open)

> "This is a fact knowledge layer. It ingests PDFs, pulls out numeric facts
> with their full context — the period they cover, the basis, and the
> document's publication date — and then decides, deterministically, whether
> facts across documents corroborate, contradict, or can be reconciled by
> context."

On screen: the **Documents** tab, showing the six starter documents — three
Delhivery filings and three India-macro reports — each with a page count, a
**claim count**, a vintage date, and status `ready`.

---

## 0:20–0:50 — A new PDF, ingested live (Documents tab)

> "It's not limited to those six. Here's a document it has never seen — the
> RBI's April 2025 Monetary Policy Statement."

- Click **Choose file**, pick `demo-datasets/rbi-mpc-2025-04-09.pdf`, click **Upload**.
- The row appears immediately with status `pending → parsing → extracting`.

> "It parses the PDF, scores each chunk for fact density, and sends only the
> highest-value chunks to the extractor."

**This upload only finishes if `GEMINI_API_KEY` is set in `.env` and the free
tier isn't rate-limited.** If it sits on `extracting`, don't wait on camera —
say *"extraction runs in the background against the same pipeline"* and move
to the Findings tab. Do **not** re-run `demo_seed.py` after uploading if you
want the row to stay visible; re-running it rebuilds the database from
scratch.

---

## 0:50–2:45 — The four required cases (Findings tab)

Click **Findings**. Use the filter chips at the top to walk each verdict.

### Case 1 — Corroboration (chip: "Corroborated")

Expand the Delhivery FY24 revenue card.

> "Same fact, two documents, expressed differently. The annual report states
> revenue of ₹81,415.38 **million**. The Q4 earnings deck states ₹8,142
> **crore**. The system normalizes the units — ₹81,415.38 million is ₹8,141.5
> crore — and the values agree within rounding tolerance."

Point at: the two evidence quotes with page numbers, and `rule fired:
unit_normalized_match`.

### Case 2 — Genuine contradiction (chip: "Contradicted")

Expand the India current-account-deficit card.

> "India's current account deficit as a percent of GDP for 2024-25. The RBI's
> Annual Report reports **-1.3%**, marked provisional. The IMF's Article IV
> report says **-0.6%**. Both agree almost exactly on the prior year, so
> this isn't a definitional mismatch — and the gap is larger than 50%, so
> the vintage difference between the two publications isn't allowed to
> explain it away."

Point at: the two evidence quotes, `rule fired: large_gap_despite_vintage_gap`,
and — if present — the LLM explanation line and any override note showing
"the rules said X, the model said Y, here's why."

### Case 3 — Reconciled by context (chip: "Reconciled")

Expand the prospectus total-income card.

> "This looks like a contradiction: total income of ₹49,114 million against a
> full-year figure that's much larger. But the prospectus figure covers the
> **nine months ended December 31, 2021** — it's a period-axis mismatch, not
> a disagreement. The system names the axis: `period`."

Point at: the two evidence quotes, `axis: period`, `rule fired:
envelope_difference_explains`.

### Case 4 — An extraction/reasoning failure (Failures tab)

Click **Failures**.

> "Case four: things the system could not place confidently, surfaced
> deliberately instead of dropped. RBI's wide multi-year appendix tables lose
> their column-to-year alignment when flattened to text, so a bare number
> can't be safely attributed to a year — those extractions are rejected here
> with the reason. Every evidence quote is also verified against the source
> chunk; a quote that can't be located is rejected rather than trusted."

Point at a row or two with `reason` and `kind`.

---

## 2:45–3:00 — Close

> "The comparison logic is pure, deterministic, and unit-tested against these
> exact real numbers — the LLM only extracts claims and writes explanations,
> it never decides a verdict. Every response is cached to disk and committed,
> so all four cases reproduce from a clone with no API key."

Stop recording.

---

## Notes for the recorder

- If a Findings card's `explanation` line is missing, that relation was
  persisted but not yet given an LLM explanation (explanation budget); the
  rule verdict and evidence are still shown and are the point. You can run
  `POST /api/reconcile` (e.g. `curl -X POST http://127.0.0.1:8000/api/reconcile`)
  once beforehand to fill explanations.
- The four cards are found fastest via the filter chips; within a verdict,
  the relevant card is the one whose measure is revenue / current account /
  total income.
- Keep it under 3:00. If you run long, cut the Case 4 narration to one
  sentence.
