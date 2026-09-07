# Fact Knowledge Layer — Design

**Date:** 2026-09-07
**Status:** Draft for review
**Assignment:** Superjoin VIT 2026 · Engineering Intern

---

## 1. Problem

Build a system that ingests PDFs, extracts grounded facts, and identifies when facts across documents **corroborate**, **contradict**, or **can be reconciled through context**. It must accept new PDFs through an API or UI and must not depend on hard-coded facts, filenames, or document-specific rules.

The assignment states plainly that a graph database or visualization alone is not the solution. The interesting work is how facts are *discovered, grounded, compared, and explained*. This design puts the comparison logic at the centre and treats storage as an implementation detail.

### 1.1 Starter data

Two independent three-document sets, all with extractable text layers (verified: 2,282–6,112 chars/page across 511 pages; no OCR required).

| Set | Documents | Nature of overlap |
|---|---|---|
| `delhivery/` | Prospectus 2022 (100pg), Annual Report FY24 (100pg), Q4 FY24 earnings deck (27pg) | Same company, three disclosure formats, two-year gap, mixed units (₹ million vs ₹ crore) |
| `india-macroeconomy/` | Economic Survey 2024-25 (89pg), RBI Annual Report 2024-25 (100pg), IMF Article IV 2025 (95pg) | Same economy, three publishers, different data vintages and projection bases |

A confirmed example already located in the data: the prospectus reports `Total income 49,114.06` for the **nine months ended December 31, 2021**, in a table adjacent to full-financial-year figures, denominated in **₹ millions**, while the annual report and earnings deck use fiscal years and ₹ crore. Same metric, incompatible context envelopes. This is the required case 3.

---

## 2. Core idea: the context envelope

A fact is not `(subject, predicate, value)`. Stored that way, `Total income = 49,114.06` and `Total income = 8,142` look like a flat contradiction when they are both true.

A fact is a **claim wrapped in a context envelope**:

```
Claim = (subject, measure, value, unit)  +  Envelope(period, basis, vintage)
```

- **period** — the time range the value covers ("FY2024", "9M ended 2021-12-31")
- **basis** — scope qualifiers: consolidated/standalone, restated/reported, actual/projected, provisional/final, including/excluding one-offs
- **vintage** — when the *document* was published, which is distinct from the period the value covers. Two documents can report the same period and disagree because one is a later revision of the other.

Comparison then becomes two stages:

1. **Alignment** — do these two claims talk about the same thing? (`subject_key` + `measure_key` match)
2. **Reconciliation** — given they align, normalize units, compare values, and diff the envelopes. If the values differ, ask whether an envelope difference *explains* the difference.

The four required cases are not four features. They are **four outcomes of one algorithm**:

| Envelope | Values | Verdict |
|---|---|---|
| identical | agree after unit normalization | `CORROBORATES` |
| identical | disagree | `CONTRADICTS` |
| differ on a known axis | disagree | `RECONCILED_BY_CONTEXT` (axis named) |
| incomplete or incommensurable | — | `INSUFFICIENT_CONTEXT` (case 4) |

This makes case 4 a first-class output rather than an embarrassed footnote. Facts the system cannot confidently place are surfaced in a dedicated view, not silently dropped or silently guessed.

### 2.1 Reconciliation axes

| Axis | Meaning | Example |
|---|---|---|
| `period` | different or nested time ranges | 9M ended Dec-2021 vs FY2021 |
| `unit` | convertible denominations | ₹ million vs ₹ crore; % vs bps |
| `scope` | basis flags differ | consolidated vs standalone |
| `vintage` | same period, later publication | RBI provisional vs revised estimate |
| `subject_granularity` | entity vs group | "Delhivery Limited" vs "Delhivery Group" |

Axes are data, not code branches — new basis flags appear in the `basis` JSON as documents introduce them, which is the "schema evolves dynamically" brownie point in its cheapest honest form.

---

## 3. Architecture

```
PDF upload
   │
   ├─[1] PARSE       pymupdf → per-page text + table blocks,
   │                 retaining (page_no, char_start, char_end, bbox)
   │
   ├─[2] CHUNK+SCORE chunks = prose blocks and table blocks.
   │                 Each scored for fact density. Priority queue.
   │
   ├─[3] EXTRACT     batched LLM call → Claim JSON with envelope fields.
   │                 Cached by content_hash. Budget-bounded.
   │
   ├─[4] ALIGN       group claims by (subject_key, measure_key).
   │                 Deterministic. No LLM, no embeddings.
   │
   ├─[5] RECONCILE   pairwise within group: normalize → compare → diff
   │                 envelopes → verdict + rule_fired. Pure functions.
   │
   ├─[6] EXPLAIN     batched LLM call, ONLY for CONTRADICTS and
   │                 RECONCILED_BY_CONTEXT pairs. Writes prose reasoning,
   │                 may override the verdict with a logged reason.
   │
   └─[7] SERVE       SQLite → FastAPI → single-page UI
```

Steps 4 and 5 are where the assignment's value sits and are **entirely deterministic** — a grader can be shown the exact rule that fired. Steps 1–3 are plumbing that must be correct but is not the differentiator.

### 3.1 Why deterministic reconciliation, not an LLM judge

An LLM-as-judge over every candidate pair is O(n²) in API calls. On a free-tier quota (~10–15 RPM) that is unrunnable, uncacheable, and produces reasoning that cannot be falsified. Instead:

- **Code decides the verdict.** Auditable, instant, free, deterministic.
- **The LLM writes the explanation**, for the small subset of pairs that are interesting — dozens, not thousands — and may override a verdict *with a stated reason that is stored*.

Every override is logged to `llm_overrode` / `override_reason` columns. The disagreement rate between the rules and the model is itself a reportable result and goes in the README.

### 3.2 Cost and rate-limit strategy

The free tier is the binding constraint, and designing around it produces the architecture we would want regardless:

| Constraint | Response | Brownie point satisfied |
|---|---|---|
| ~10–15 RPM | Token-bucket limiter, backoff on 429, concurrency ≤ 2 | — |
| 511 pages of corpus | Chunks scored, processed by descending fact-density under a configurable call budget | large PDFs without performance issues |
| Re-runs must be cheap | Content-hash cache keyed on `(chunk_hash, prompt_version)`, persisted to `data/cache/` and **committed to the repo** | graders reproduce with no API key |
| New document added | Only its own chunks extract; reconciliation runs against existing claims | incremental ingest without rebuild |

---

## 4. Data model

SQLite with JSON columns. Chosen over a graph database deliberately: the assignment says a graph DB alone is not the solution, and our relationships are pairwise within small aligned groups, not multi-hop traversals. SQLite keeps setup to `pip install` with no service to run.

### `documents`
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | uuid4 |
| `filename` | TEXT | as uploaded |
| `sha256` | TEXT UNIQUE | dedupe on re-upload |
| `page_count` | INTEGER | |
| `title` | TEXT | extracted, nullable |
| `publisher` | TEXT | extracted, nullable |
| `doc_date` | TEXT | ISO date; the **vintage**. Extracted, nullable |
| `uploaded_at` | TEXT | ISO timestamp |
| `status` | TEXT | `pending` / `parsing` / `extracting` / `ready` / `failed` |

### `chunks`
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | |
| `doc_id` | TEXT FK | |
| `page_no` | INTEGER | 1-indexed PDF page |
| `kind` | TEXT | `prose` \| `table` |
| `text` | TEXT | |
| `char_start` | INTEGER | offset within page text |
| `content_hash` | TEXT | sha256 of normalized text — the cache key |
| `score` | REAL | fact-density score |
| `extracted` | INTEGER | 0/1 |

### `claims`
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | |
| `doc_id`, `chunk_id` | TEXT FK | |
| `subject` | TEXT | as written |
| `subject_key` | TEXT | normalized slug |
| `measure` | TEXT | as written |
| `measure_key` | TEXT | canonical slug, e.g. `revenue_from_operations` |
| `value_type` | TEXT | `numeric` \| `text` \| `date` \| `boolean` |
| `value_num` | REAL | nullable |
| `value_text` | TEXT | nullable |
| `unit_raw` | TEXT | "₹ million" |
| `unit_dim` | TEXT | dimension: `currency_inr` \| `percent` \| `count` \| `ratio` \| null |
| `unit_scale` | REAL | multiplier to base unit (₹ million → 1e6) |
| `period_start` | TEXT | ISO date, nullable |
| `period_end` | TEXT | ISO date, nullable |
| `period_label` | TEXT | as written |
| `basis` | TEXT (JSON) | `{"consolidated": true, "restated": false, ...}` |
| `evidence_page` | INTEGER | |
| `evidence_quote` | TEXT | verbatim span from the source |
| `evidence_char_start` | INTEGER | offset into page text, for highlighting |
| `confidence` | REAL | 0–1, model-reported |
| `extractor_version` | TEXT | prompt version, for cache invalidation |

**Evidence grounding is validated, not trusted.** After extraction, every `evidence_quote` is checked to actually occur in its chunk text. Claims whose quote cannot be located are rejected and logged as extraction failures. This is the guard against the model inventing a citation.

### `relations`
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | |
| `claim_a_id`, `claim_b_id` | TEXT FK | ordered by claim id for stability |
| `rule_verdict` | TEXT | output of the deterministic gates (Section 5), never overwritten |
| `final_verdict` | TEXT | `rule_verdict` unless the LLM explanation pass overrides it; **this is what the UI and API show as "the verdict"** |
| `axis` | TEXT | nullable; one of the reconciliation axes |
| `rule_fired` | TEXT | name of the deterministic rule that produced `rule_verdict` |
| `delta_pct` | REAL | nullable, relative difference after normalization |
| `explanation` | TEXT | nullable; LLM-written prose |
| `llm_overrode` | INTEGER | 0/1; true whenever `final_verdict != rule_verdict` |
| `override_reason` | TEXT | nullable; required whenever `llm_overrode = 1` |

Splitting `verdict` into `rule_verdict` and `final_verdict` keeps the deterministic
output auditable even after an override — the UI can show "the rules said X,
the model said Y, here's why" as a first-class comparison rather than losing
the rule's original answer. `GET /api/relations` (Section 7) filters and
displays by `final_verdict`; `rule_verdict` is shown alongside it whenever
they differ.

### `measure_registry`
| Column | Type | Notes |
|---|---|---|
| `measure_key` | TEXT PK | |
| `label` | TEXT | human-readable |
| `aliases` | TEXT (JSON) | surface forms seen |
| `unit_dim` | TEXT | expected dimension |
| `seen_count` | INTEGER | |

The registry is fed back into the extraction prompt so slugs **converge** rather than drift across documents. This is the mechanism that makes alignment work without embeddings, and it grows as new measures appear.

---

## 5. The reconciliation algorithm

Pure functions over two claims. No I/O, no LLM. Fully unit-testable — this is where the test suite's weight goes.

```
reconcile(a, b) -> Relation

GATE 0  Incommensurable units
        a.unit_dim != b.unit_dim  ->  INSUFFICIENT_CONTEXT, axis=unit,
                                      rule=incommensurable_units
        (comparing a percentage to a rupee amount is not a contradiction)

GATE 1  Incomplete envelope
        either side missing period_end when value_type == numeric
        ->  INSUFFICIENT_CONTEXT, axis=null, rule=incomplete_envelope

NORMALIZE (runs after the gates above, before GATE 2)
        va = a.value_num * a.unit_scale
        vb = b.value_num * b.unit_scale
        agree = |va - vb| / max(|va|,|vb|) <= tolerance
        tolerance = 0.01, widened to 0.05 when either side shows
                    rounding (<= 3 significant figures)

GATE 2  Identical envelopes  (period_rel == same AND basis_differs == false)
        agree                              ->  CORROBORATES
                                               rule = exact_envelope_match, or
                                               unit_normalized_match if units differed
        not agree, vintage gap >= 90d,
        delta_pct <= MAGNITUDE_CAP (0.5)   ->  RECONCILED_BY_CONTEXT, axis=vintage
                                               rule = later_vintage_revision
        not agree, vintage gap >= 90d,
        delta_pct >  MAGNITUDE_CAP (0.5)   ->  CONTRADICTS
                                               rule = large_gap_despite_vintage_gap
        not agree, no vintage gap          ->  CONTRADICTS
                                               rule = same_envelope_value_mismatch

GATE 3  Envelopes differ
        primary_axis = first of [period, scope, vintage, subject_granularity]
                       that actually differs   (period is most explanatory)
        agree      ->  RECONCILED_BY_CONTEXT,
                       rule = coincident_values_differing_envelope
                       (values match but contexts do not — reported, not
                        silently upgraded to corroboration)
        not agree  ->  RECONCILED_BY_CONTEXT, axis=primary_axis,
                       rule = envelope_difference_explains
```

**The `MAGNITUDE_CAP` on vintage-explained differences (0.5, i.e. 50% relative
difference)** exists because "later publication date" is not by itself proof
that a later figure is a *revision* of an earlier one — it is equally
consistent with two institutions independently estimating the same thing and
disagreeing. A small gap across a vintage boundary reads as an ordinary data
revision; a large one, especially a **directional** one (deficit widening in
one source, narrowing in the other), is the kind of disagreement a rule
should surface rather than paper over. Above the cap, the gate emits
`CONTRADICTS` by default — conservative — and leaves it to the LLM
explanation pass (Section 6.2) to make the case for `RECONCILED_BY_CONTEXT`
if it can, with the override stored in `final_verdict` alongside the
unmodified `rule_verdict`. This was found empirically: probing the macro
dataset surfaced exactly this shape of case (Section 11, case 2), where a
pure vintage-gap rule without a magnitude check would have silently explained
away a real disagreement.

**Basis comparison** is defined precisely to avoid every pair looking different. `basis_differs(a, b)` considers **only keys present on both sides**, and is true when any shared key holds a different value. A key present on one side only means *unstated*, not *contradicted* — documents routinely omit qualifiers they consider obvious. The one exception: if exactly one side asserts `restated: true` or `projected: true`, that counts as a difference on the `scope` axis, because those two qualifiers change the meaning of a figure even when the other side is silent.

**Subject granularity** is detected structurally, not semantically: two `subject_key` values sit on this axis when one is a strict token-prefix of the other after slug normalization (`delhivery_limited` vs `delhivery_limited_group`). Anything looser is left alone — alignment already requires exact `subject_key` equality, so this axis only annotates deliberately linked groups and is the least-used axis in practice.

**Period comparison** (`reason/periods.py`) returns one of `same`, `nested`, `overlapping`, `disjoint`, `unknown`. Handles Indian fiscal years (FY2024 = 2023-04-01 to 2024-03-31), quarters, "nine months ended <date>", calendar years, and bare years.

**Unit handling** (`reason/units.py`) parses ₹/INR/Rs with scale words (thousand, lakh, million, crore, billion), percent, bps, counts, and ratios, returning `(unit_dim, scale)`. Unknown units yield `unit_dim = None`, which GATE 0 treats as incommensurable — deliberately conservative.

### 5.1 Pair volume

Reconciliation is pairwise *within* an aligned group, not across the corpus. Groups are small (typically 2–8 claims), so total pairs stay in the low thousands even at corpus scale, and each pair costs microseconds.

---

## 6. LLM interface

Exactly **two** call sites. Both batched, both cached, both offline-replayable.

### 6.1 Extraction (`extract/prompts.py::EXTRACTION_PROMPT`)

Input: several chunks + the current measure registry (top-N by `seen_count`).
Output: strict JSON array of claim objects.

Requirements encoded in the prompt:

- `evidence_quote` must be copied **verbatim** from the chunk (validated after)
- prefer an existing `measure_key` from the registry; mint a new snake_case slug only when nothing fits
- `basis` keys are free-form — emit whatever qualifiers the text states
- omit a field rather than guess it; `confidence` must reflect real uncertainty
- extract semantic facts (directorships, addresses, ratings, statuses) as well as numeric ones

Malformed JSON is repaired once by a bounded retry, then the batch is logged as a failure and skipped. Ingest never crashes on a bad model response.

### 6.2 Explanation (`reason/explain.py`)

Input: batch of relations with `rule_verdict` in `{CONTRADICTS, RECONCILED_BY_CONTEXT}`, each with both claims and both evidence quotes.
Output: per-relation `{explanation, override?, override_reason?}`.

The model is explicitly told the rule that fired and invited to disagree with it. Overrides are stored, never silently applied.

### 6.3 Provider

Gemini 2.5 Flash on the free tier (1M context suits batching). The client is a thin interface (`extract/llm.py`) with a single `complete_json()` method, so swapping to Groq or a local model is a one-file change. API key via `.env`, never committed.

**Offline mode:** if no key is present, the system runs entirely from `data/cache/`. Graders can process the starter documents and see all four cases with no credentials.

---

## 7. API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | multipart upload; returns `{doc_id}`; ingest runs in background |
| `GET` | `/api/documents` | list with status and claim counts |
| `GET` | `/api/documents/{id}` | detail + ingest progress |
| `GET` | `/api/claims` | filter by `doc_id`, `subject_key`, `measure_key`, `q` |
| `GET` | `/api/groups` | aligned groups with per-verdict relation counts |
| `GET` | `/api/groups/{subject_key}/{measure_key}` | claims + relations + evidence |
| `GET` | `/api/relations` | filter by `final_verdict`, `axis` — **this is how the four cases are demoed** |
| `GET` | `/api/evidence/{claim_id}` | page text with the quote span marked |
| `GET` | `/api/failures` | rejected extractions, low-confidence claims, `INSUFFICIENT_CONTEXT` relations |
| `POST` | `/api/reconcile` | re-run steps 4–6 without re-extracting |

## 8. UI

One HTML page, vanilla JS, no build step — `uvicorn` serves it directly.

- **Documents** — drag-drop upload, list, live ingest progress
- **Facts** — searchable claim table; each row shows value, envelope, source
- **Findings** — the primary screen. Filter chips for `Corroborated` / `Contradicted` / `Reconciled` / `Needs review`. Each row expands to: side-by-side evidence quotes with the span highlighted, an envelope-diff table showing which axis differs, the `rule_fired`, and the explanation.
- **Failures** — case 4, presented deliberately rather than hidden

The Findings expansion is the screen the demo video is built around, because it shows evidence, envelope, rule, and reasoning in one frame.

---

## 9. File structure

```
app/
  main.py              FastAPI app + route registration
  config.py            settings, env loading
  db.py                schema DDL, connection, migrations
  models.py            dataclasses for Document/Chunk/Claim/Relation
  ingest/
    parse.py           pymupdf -> pages, table blocks, offsets
    chunk.py           blocks -> Chunk with content_hash
    score.py           fact-density scoring, priority ordering
    pipeline.py        orchestration: parse -> chunk -> extract -> reconcile
  extract/
    llm.py             provider client, rate limiter, disk cache
    prompts.py         EXTRACTION_PROMPT, EXPLANATION_PROMPT, versions
    claims.py          batching, JSON validation, evidence verification
    registry.py        measure_key registry read/write
  reason/
    units.py           unit parsing and conversion
    periods.py         period parsing and comparison
    align.py           grouping into aligned sets
    reconcile.py       the gates; returns Relation
    explain.py         LLM explanation pass over interesting relations
  static/
    index.html, app.js, style.css
tests/
  test_units.py, test_periods.py, test_reconcile.py    <- the weight
  test_claims.py       evidence-verification and JSON-repair tests
  test_api.py          smoke tests against a seeded DB
data/
  cache/               committed LLM responses
  facts.db             gitignored
```

Fourteen focused source files. Nothing over roughly 200 lines.

## 10. Testing

Reconciliation, units, and periods are pure functions and get thorough table-driven unit tests — including the confirmed real cases from the starter data. Extraction is tested against **cached fixtures**, so the suite never makes a network call and runs in CI without a key. API tests run against a seeded database.

Deliberately not tested: LLM output quality. That is measured by inspection and reported honestly in the README, not asserted.

---

## 11. Demonstrating the four required cases

All four cases below were located by direct probe of the extracted PDF text
(page-cited) before implementation began, so the build targets confirmed
evidence rather than a hoped-for shape of data.

**1. Corroborated across documents, expressed differently.**
Delhivery FY24 consolidated revenue from operations.
- Annual Report FY24: *"revenue from operations on consolidated basis for
  FY24 stood at ₹81,415.38 million"*
- Q4 FY24 earnings presentation (₹ Cr table): *"Total revenue from
  customers ... FY24: 8,142"*
- ₹81,415.38 million ÷ 10 = ₹8,141.54 crore ≈ ₹8,142 Cr — agrees within
  rounding tolerance. Expected verdict: `CORROBORATES`,
  `rule = unit_normalized_match`.
- A second, unit-free example: RBI (*"real GDP growth moderated to 6.5 per
  cent in 2024-25"*) and IMF (*"India's real GDP grew by 6.5 percent in
  FY2024/25"*) — exact match, no conversion needed. Expected verdict:
  `CORROBORATES`, `rule = exact_envelope_match`.

**2. A genuine or likely contradiction.**
India's Current Account Deficit as % of GDP, FY2024-25 — both sources agree
on the prior year, then diverge directionally on the current one:

| Source | FY2023-24 | FY2024-25 |
|---|---|---|
| RBI Annual Report (table, marked "(P)" provisional) | -0.7% | **-1.3%** (widening) |
| IMF Article IV (own balance-of-payments table) | -0.7% | **-0.6%** (narrowing) |

Neither document states a differing scope or definition; the FY23-24 anchor
matches almost exactly (-0.7% both), which rules out a systematic
definitional gap. The vintage gap between the two publications is large
(RBI ~May 2025, IMF ~November 2025), but the delta (`|-1.3 - -0.6| / 1.3` ≈
0.54) exceeds `MAGNITUDE_CAP`, and the direction of change is opposite
between sources. Expected verdict: `rule_verdict = CONTRADICTS`,
`rule = large_gap_despite_vintage_gap` — a direct exercise of the Section 5
magnitude cap this probe motivated. Whether the LLM explanation pass can
argue this back to `RECONCILED_BY_CONTEXT` (e.g. citing the "(P)" provisional
marker) or leaves it as a genuine contradiction is itself worth showing in
the demo as a rules-vs-model disagreement.

**3. An apparent contradiction explained by context.** Two confirmed examples:
- *Period axis:* the 2022 prospectus reports `Total income 49,114.06` for the
  nine months ended 2021-12-31, in ₹ million, in a table adjacent to
  full-financial-year figures.
- *Vintage axis:* the Economic Survey states *"India's real GDP is estimated
  to grow by 6.4 per cent in FY25"*, explicitly labeled a **First Advance
  Estimate** (published before the fiscal year ends), against RBI/IMF's 6.5%
  from the later Provisional Estimate — a 0.1pp gap fully explained once the
  vintage label is read. This mirrors the assignment's own inspiration
  example almost exactly.

**4. An extraction or reasoning failure.**
Found directly while probing, not manufactured: RBI's wide multi-year
appendix tables (e.g. six columns spanning `2020-21` through `2024-25 (P)`)
lose their column-to-year alignment once linearized into plain text —
confidently attributing a bare value like `-1.3` to `2024-25` required
manually cross-referencing two separate tables' worth of context. A
chunk-based extraction pass over raw page text is exactly liable to
misattribute such a cell to the wrong year. This is the concrete case for
using `pymupdf`'s structured `find_tables()` output (Section 9) for table
chunks instead of linear text, and for surfacing low-confidence
column-attribution claims via `/api/failures` rather than asserting them.

## 12. Scope boundaries (YAGNI)

Explicitly **not** building, given the 1–2 day budget:

- Embeddings or a vector store — the measure registry handles alignment
- A graph database — pairwise relations within small groups do not need one
- Entity resolution beyond slug normalization
- Authentication, multi-tenancy, or deployment tooling
- OCR — verified unnecessary for this corpus
- Multi-hop or transitive inference across relations
- A JS build pipeline

Each appears in the README's "Next Steps" with the reason it was cut.

## 13. Scaling beyond the starter corpus

The design targets the starter corpus (6 PDFs, ~511 pages) directly, but the
pipeline stages differ sharply in how they behave as document count grows.
This section records which stages already scale, which break first, and the
fix for each — so scale is an explicit design decision, not an afterthought.

### 13.1 Already scale-safe as designed

- **Parse/chunk** — pure CPU work, linear in page count, independent per PDF.
- **Alignment** — grouping by `(subject_key, measure_key)` is an indexed
  lookup, O(n) regardless of corpus size, given a DB index on those columns.
- **Incremental ingest** — a new PDF extracts only its own chunks. Its claims
  join existing groups; relations already computed for existing pairs are
  never recomputed. Cost per new document is bounded by *that document's*
  claim count, not the corpus size — the difference between "PDF #501 costs
  O(1 document)" and "PDF #501 triggers a full recompute."
- **Content-hash cache** — boilerplate repeated across similar filings
  (disclaimers, standard headers) becomes a cache hit after the first
  occurrence, so more documents of a similar type get *cheaper* per document,
  not more expensive.

### 13.2 Where it breaks first, and the fix

**Group-size blowup in reconciliation.** Reconciliation is pairwise *within*
a group, sized in this design for the starter corpus's 2–8 claims per group.
That assumption fails if a common measure (e.g. "GDP growth rate") accumulates
claims from hundreds of documents — naive pairwise comparison is O(k²) and
hits ~125,000 pairs at k=500.

*Fix:* bucket claims within a group by exact `(period, basis)` before
comparing. Reconcile one representative claim per bucket against other
buckets, then fan the resulting verdict out to every claim sharing that
bucket. This turns O(k²) into O(distinct_envelopes²), and distinct envelopes
grow far slower than raw claim count as more documents repeat the same fiscal
years and bases. Not implemented for the starter corpus — group sizes there
never exceed single digits — but isolated entirely inside `reason/align.py`,
so it is a targeted addition, not a redesign, when group sizes warrant it.

**SQLite's single-writer concurrency.** Sufficient for one background ingest
job at a time, which covers the assignment. It becomes a bottleneck only
under concurrent uploads from multiple users. Mitigation: all DB access is
already isolated behind `db.py` (Section 9), so swapping the connection
string to Postgres is a one-file change with no call-site edits elsewhere.
Not undertaken now — premature for this scope.

**LLM rate limits — the one wall architecture cannot remove.** More documents
means more extraction calls, linearly, against a fixed quota. This is why the
design already scores and prioritizes chunks (spend budget on the
highest-density content first), caches aggressively (no repeated spend on
re-ingest or re-run), and keeps the call budget configurable (a large batch
can be throttled to run over hours instead of live). Beyond the free tier,
the only real fix is a paid tier or a cheaper/local model for extraction —
a configuration change in `extract/llm.py`, not an architecture change,
because of the thin single-method provider interface (Section 6.3).

**Storage growth.** Chunk and claim text is heavy but SQLite comfortably
handles low-single-digit-GB databases; indexes on `content_hash`,
`subject_key`, and `measure_key` keep lookups fast well past hundreds of
documents. Not a near-term concern.

### 13.3 What this means for the build

None of the above is built for this submission — the starter corpus never
exercises these limits, and building for scale we cannot demonstrate would
violate the YAGNI boundary in Section 12. This section exists so the
trade-off is a stated decision in the README rather than a gap a grader has
to infer.

## 14. Known risks

| Risk | Mitigation |
|---|---|
| `measure_key` drift across documents breaks alignment | Registry fed back into every extraction prompt; a review of the registry after ingest is part of the build |
| Free-tier rate limits make ingest slow for a live demo | Cache committed; the video demonstrates upload of a smaller PDF while the starter corpus is pre-cached |
| Table structure lost by text extraction, corrupting values | Tables chunked separately with structure preserved; mis-associations become case 4 material |
| Model invents evidence quotes | Every quote verified against chunk text; unverifiable claims rejected and logged |
| A vintage-gap rule over-explains a real disagreement as "just a later revision" | `MAGNITUDE_CAP` in GATE 2 (Section 5) forces large or directional gaps to `CONTRADICTS` by default; confirmed necessary by the CAD example in Section 11 case 2 |
