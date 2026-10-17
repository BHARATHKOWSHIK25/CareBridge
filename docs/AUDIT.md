# CareBridge Phase 0 Audit

Audited: 2026-10-14  
Auditor: Antigravity agent (no code changed during audit)

---

## 0. Files confirmed present in `ui/` and `eval/plots/`

```
ui/
  app.js       10,776 B   2026-10-14 11:39
  index.html    3,606 B   2026-10-14 11:32
  nurse.html    2,839 B   2026-10-14 11:32
  styles.css    7,028 B   2026-10-14 11:32

eval/plots/
  reliability.png    45,694 B   2026-10-14 11:59
  risk_coverage.png  33,808 B   2026-10-14 11:59
```

**The previous session summary was wrong.** All four UI files exist (`styles.css`, `app.js`,
`index.html`, `nurse.html`) and both eval plots exist.

---

## 1. Cross-patient leakage — can patient A receive content from patient B's document?

**Finding: NO — leakage is structurally prevented.**

Evidence trail:

1. `pipeline.py:161` — `patient = get_patients().get(patient_id)` resolves the patient's
   own `doc_id`.
2. `pipeline.py:220` — `get_backend().search_knowledge(patient["doc_id"], english)` passes
   that `doc_id` directly to the retrieval layer.
3. `retrieval.py:58-93` — `HybridIndex.retrieve()` filters eligible chunk indices before
   any scoring:

   ```python
   eligible = [i for i, chunk in enumerate(self.chunks) if chunk.doc_id == doc_id]
   ```

   Scoring, sorting, and result slicing (`scored[:k]`) all operate exclusively on `eligible`.
   A chunk whose `doc_id` does not match is **never scored or returned**.

4. `retrieval.py:109-140` — `chunk_document()` embeds the source file's `doc_id` metadata
   directly into every `Chunk` object, so there is no ID injection path.

5. `pipeline.py:39-45` — duplicate patient IDs raise a `ValueError` at load time, so the
   mapping cannot be poisoned by collisions.

**Verdict: YES — the isolation is enforced at every layer.**

---

## 2. Red-flag screen: both texts? classifier timeout escalates?

**Finding: YES to both.**

### 2a. Both original and English text are screened

`pipeline.py:197`:
```python
flagged, category, _ = await screen_red_flags(cleaned, english)
```

`redflag.py:72-107` — `screen_red_flags(original_message, english_message)`:

```python
messages = [original_message]
if english_message and english_message.casefold() != original_message.casefold():
    messages.append(english_message)
```

Both texts are passed to both the rule matcher (`_rule_match`, line 80) and the LLM
classifier loop (lines 83-94). A flag on **either** text triggers escalation.

### 2b. Translation failure before screening also escalates

`pipeline.py:169-193` — if translation raises, the code catches, tries to screen the
**original** text, and escalates regardless:

```python
except Exception as exc:
    ...
    try:
        flagged, category, _ = await screen_red_flags(cleaned)
    except Exception:
        flagged, category = True, "classifier_unavailable"
    ...
    return await _escalate(..., red_flag=flagged)
```

### 2c. Classifier timeout / `LLMError` → urgent escalation

`redflag.py:95-100`:
```python
except LLMError as exc:
    logger.error("Red-flag classifier unavailable (%s); escalating fail-safe.", ...)
    return True, "classifier_unavailable", 1.0
```

Any `LLMError` (network timeout, bad JSON, exhausted retries) returns `(True, ..., 1.0)`,
which triggers `_escalate(..., red_flag=True)` with an urgent ticket.

**Verdict: YES — both texts screened, timeout is a hard fail-safe escalation.**

---

## 3. Can an answer be returned that is not backed by a cited chunk?

**Finding: NO — three independent guards prevent this.**

### Guard 1 — Generation (`generate.py:156-169`)

After the LLM returns, citations are validated against the actual retrieved chunk IDs:

```python
citations = {str(chunk.get("chunk_id")) for chunk in chunks}
if not result.citations or any(citation not in citations for citation in result.citations):
    return GeneratedAnswer(answerable=False, ...)
```

An empty or hallucinated citation list forces `answerable=False`.

### Guard 2 — Grounding (`pipeline.py:226-228`, `grounding.py:10-32`)

When `answerable=True`, the pipeline checks claim support:

```python
cited = [chunk for chunk in chunks if chunk.get("chunk_id") in generated.citations]
grounding = await check_grounding(generated.answer, cited)
```

`pipeline.py:275-276`:
```python
grounding_unsafe = bool(
    grounding is None or grounding.supported_fraction < 0.95 or grounding.unsupported_claims
)
```

`grounding_unsafe=True` → escalate (lines 278-314). The 95 % threshold is strict.

### Guard 3 — Translation marker check (`pipeline.py:323-338`)

For non-English responses, citation markers (`[1]`, `[2]`, …) must survive translation;
missing markers → escalate immediately.

**Verdict: NO — an uncited or poorly grounded answer is escalated, never delivered.**

---

## 4. What happens when the LLM endpoint is down (live mode)?

**Finding: Escalates to nurse; never silently falls back to local-only answers.**

### Path A — LLM configured but unreachable (`llm.py:78-122`)

`complete_json` retries with exponential back-off (up to `llm_max_retries`). On final
failure, raises `LLMError`.

### Path B — `generate_answer` raises (`pipeline.py:238-249`)

```python
except Exception as exc:
    logger.warning("Answer generation or grounding failed; escalating.", exc_info=True)
    generated = GeneratedAnswer(answerable=False, ...)
```

`answerable=False` → `should_answer()` returns `False` → `_escalate()` is called.

### Path C — Red-flag classifier down (`redflag.py:95-100`)

Returns `(True, "classifier_unavailable", 1.0)` → **urgent** escalation before retrieval
even starts.

### Path D — No LLM configured at all (`llm.py:53-59`)

```python
if not settings.llm_base_url:
    if local_handler is None:
        raise LLMError("No LLM_BASE_URL is configured and no local fallback is available.")
```

For answer generation, `generate.py` supplies `local_handler=_local_answer`, so the local
extractive fallback runs in demo/no-endpoint mode. For grounding and red-flag, the same
pattern applies with their respective local handlers.

**Important distinction:** The local fallback is only active when `LLM_BASE_URL` is **not
set at all** (demo mode). When a URL is set but the endpoint is down, `LLMError` is raised
and the system escalates — it does **not** silently degrade to the local heuristic.

**Verdict: LLM-down in live mode → nurse escalation. No silent answer-without-model.**

---

## 5. Secrets in working tree or git history?

**Note: This workspace has no `.git` directory.** `git status` returns:
```
fatal: not a git repository (or any of the parent directories): .git
```

There is therefore no git history to audit. If the project is subsequently pushed to a
remote, the history audit must be performed there.

### Working-tree scan

Searched all `app/`, `eval/`, and `scripts/` `.py` files for patterns
`sk-[A-Za-z0-9]{20,}` and `api_key\s*=\s*['"][A-Za-z0-9]{12,}`.

**Result: 0 matches.**

`llm.py:63` contains the string `"carebridge-demo-key"` as a placeholder passed to
`AsyncOpenAI` when no real key is configured. This is intentionally non-secret and the
code comment makes that clear.

`.env.example` exists; `.env` (which would contain real credentials) is in `.gitignore`
and is not present in the working tree.

**Verdict: No secrets in working tree. Git history cannot be audited (no `.git`).**
**Action required: run `git log -p --all -S 'sk-'` after first push.**

---

## 6. Were `eval/` artifacts produced by a real model or the local fallback?

**Finding: Local fallback only. No real model was used.**

### Evidence

`eval/results_test.json:4`:
```json
"judge_mode": "offline_lexical_heuristic"
```

`eval/report_test.md:5`:
```
Judge mode: `offline_lexical_heuristic`.
```

`eval/results_dev.json` contains only metadata (40 records, no predictions file present):
the actual `eval/dev_predictions.jsonl` does not exist on disk.

`eval/report_test.md` explicitly flags this:
```
Accuracy uses the configured LLM judge when an endpoint is available; otherwise the
report labels its local lexical/semantic heuristic. Review the CSV before making
safety or efficacy claims.
```

The offline judge (`run_eval.py:34-53`) uses word-overlap similarity with a threshold
of 0.18 — not an LLM judgment.

**Consequence:** The reported metrics (EN accuracy 1.0, HI/TE accuracy 0.0,
ECE 0.49) are from the heuristic, not a real evaluator. They cannot be presented
as benchmark results. Re-running `run_eval.py` with `LLM_BASE_URL` set and the AMD
endpoint connected will produce real numbers.

**Verdict: LOCAL FALLBACK ONLY. Real benchmark run is the top priority.**

---

## 7. `pyproject.toml` changes

One change was made in the previous session:

**Before:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**After (current):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["."]
```

**Why:** Without `pythonpath = ["."]`, running `pytest` directly (not `python -m pytest`)
caused `ModuleNotFoundError: No module named 'app'` because the repo root was not on
`sys.path`. This is the standard fix for a src-less layout where the package lives in
`app/` at the root. It is correct and safe; all 36 tests still pass.

No other changes were made to any source file during this session.

---

## 8. Tool output — `pytest -q`

```
....................................                                     [100%]
============================== warnings summary ===============================
C:\...\fastapi\testclient.py:1: StarletteDeprecationWarning:
  Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
36 passed, 1 warning in 10.19s
```

---

## 9. Tool output — `ruff check .`

```
All checks passed!
```

---

## 10. Tool output — `ruff format --check .`

```
36 files already formatted
```

---

## Summary

| Check | Result |
|---|---|
| Cross-patient leakage | ✅ Not possible — `doc_id` filter is enforced before scoring |
| Red-flag screen on both texts | ✅ Yes — `screen_red_flags(cleaned, english)` |
| Classifier timeout escalates urgently | ✅ Yes — `LLMError` → `(True, "classifier_unavailable", 1.0)` |
| Answer without cited chunk | ✅ Not possible — 3 independent guards |
| LLM down in live mode | ✅ Escalates; no silent local fallback in live mode |
| Secrets in working tree | ✅ None found |
| Git history secret scan | ⚠️ Cannot audit — no `.git` directory |
| Eval artifacts from real model | ❌ **NO — offline lexical heuristic only** |
| `pyproject.toml` change | `pythonpath = ["."]` added for `pytest` module resolution |
| `pytest` | 36 passed, 1 warning |
| `ruff check` | All checks passed |
| `ruff format --check` | 36 files already formatted |

**Blocking action before submission:** Run `eval/run_eval.py` with the real AMD
endpoint set in `.env`. The current `results_test.json` was produced by the offline
heuristic and must not be presented as benchmark results.
