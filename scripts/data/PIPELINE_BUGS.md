# Data-pipeline bugs (`scripts/data/master-pull.py`)

Issues found while validating the data pull by reproducing the 2025 vintage from
the 2024 workbook (start from 2024 → pull → compare against the committed 2025
files). See **[DATA_UPDATED.md](DATA_UPDATED.md)** for the corrected run guide.

---

## BUG 1 — Windows-only asyncio policy breaks the UK long-rate fetch on Linux/macOS

**Status:** fixed 2026-07-09 in `master-pull.py`.

**Location:** `master-pull.py`, the `_run()` worker thread inside the Bank of
England long-rate fetch (`ltir_uk_boe`, ~line 600).

**Symptom:**
```
Exception in thread Thread-1:
  ...
  File ".../master-pull.py", line 601, in _run
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
AttributeError: module 'asyncio' has no attribute 'WindowsProactorEventLoopPolicy'
```
The worker thread dies, `box["csv"]` is never set, the UK fetch raises, and it is
logged as a `[FAIL]` — so **`ltir_uk` is left at its existing (stale/blank)
value** while every other column refreshes.

**Root cause:** `asyncio.WindowsProactorEventLoopPolicy` exists **only on
Windows** (it is needed there for Playwright's subprocess transport). The line was
written for a Windows dev machine and called unconditionally, so it throws
`AttributeError` on the Linux cluster where the model actually runs — regardless
of whether `playwright`/Chromium are installed (they install and launch fine on
Linux; this line fails before the browser is ever used).

**Why it is easy to miss:** failed fetches fall back to existing values silently
and the pull still "succeeds" and saves a workbook. The only signals are the
thread traceback on stderr and a single `[FAIL]` line. If you skip the
completeness check (DATA.md), you ship a workbook whose UK long rate silently
stopped updating.

**Fix applied:**
```python
def _run():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    ...
```
(`sys` is already imported.) On Linux/macOS the default event-loop policy is
correct; Python 3.8+ uses `ThreadedChildWatcher`, which supports the subprocess
Playwright spawns from a non-main thread.

---

## BUG 2 — BoE UK long-rate fetch hangs forever on headless Linux (no join timeout)

**Status:** mitigated 2026-07-09 in `master-pull.py` (bounded join + daemon
thread). Underlying headless-Playwright fetch still does not succeed on the Linux
compute nodes — see "proper fix" below.

**Location:** same BoE `ltir_uk_boe` fetch, the `threading.Thread(target=_run)` /
`t.join()` just below the code in BUG 1 (~line 613).

**Symptom:** with BUG 1 fixed, the fetch *actually* drives async Playwright. On a
headless Linux node the process then **hangs indefinitely** — observed 19+ min at
~1s CPU (`futex_do_wait`), a Playwright `node` driver child alive the whole time,
but **no Chromium process ever spawned**. Because `t.join()` had no timeout, the
main thread waited forever and the entire pull never finished or wrote output.

**Root cause:** the fetch runs the **async** Playwright API inside a **worker
thread** with a fresh event loop. Launching a subprocess (Chromium) from asyncio
in a non-main thread does not reliably work on Linux, so `chromium.launch()`
never completes and the thread blocks forever; `t.join()` (no timeout) then blocks
the whole program. (Standalone, the **sync** Playwright API in the main thread
launches Chromium fine here — so it's the threaded-async design, not the browser
install, that fails.) The old Windows-only line (BUG 1) accidentally hid this by
crashing the thread instantly, which let the pull complete with `ltir_uk` skipped.

**Mitigation applied:** make the thread a daemon and bound the wait —
```python
t = threading.Thread(target=_run, daemon=True)
t.start()
t.join(timeout=150)
if t.is_alive() or "csv" not in box:
    raise RuntimeError("BoE UK long-rate fetch did not complete within 150s ...")
```
Now a hang degrades to a `[FAIL]` fallback (ltir_uk keeps existing values) and the
pull finishes instead of hanging.

**Proper fix (not yet done):** rewrite the BoE fetch to use the **synchronous**
Playwright API on the main thread (`with sync_playwright() as p: p.chromium.launch(...)`),
dropping the thread/async-loop machinery entirely. Until then, on this cluster
`ltir_uk` must be refreshed another way (run the pull on a desktop/Windows machine
where the BoE browser download works, or patch that value in by hand).

## General fragilities (not bugs, but worth knowing)

- **Silent fallback on any failed fetch.** A `[FAIL ]`/`[warn ]` column keeps its
  old values and the workbook still saves. **Always run the completeness check**
  (open the output, inspect the last 1–2 years for blank cells) before trusting a
  pull — see DATA_UPDATED.md.
- **Live sources revise.** The pull fetches current values from FRED/OECD/World
  Bank/IMF/etc., so re-running later can legitimately differ from an earlier
  vintage even with a correct pipeline. Exact reproduction is only expected
  shortly after the original pull, and only for series the providers haven't
  revised.
- **Australia short rate is a manual input** (`indata/raw/f1.1-data.csv`), not a
  live fetch — a stale CSV silently freezes `stir_au`. (Documented in DATA.md.)

---

## How these were found — reproducing a vintage (regression harness)

To validate the pipeline, copy `master-pull.py` to a throwaway script that reads
the **prior** year's workbook as its source and writes to a scratch dir, then
diff against the known-good vintage:

```python
# in the copy: point SRC at the previous vintage, OUT at a scratch dir
SRC_XLSX = os.path.join(INDATA, "DataInflShortLongConsUpdated_2024.xlsx")
OUT_XLSX = os.path.join(INDATA, "..", "indata_check", "DataInflShortLongConsUpdated_2025.xlsx")
OUT_XLSX_NO_RCONPC = os.path.join(INDATA, "..", "indata_check", "DataInflShortLongUpdated_2025.xlsx")
```
Then compare cell-by-cell, splitting results into the **≤2020 frozen block**
(must be byte-identical — it is copied from the source workbook) and the
**2021–FETCH_END re-fetched block** (should match up to source revisions). Any
≤2020 diff is a real pipeline bug; a 2021+ diff is either a bug, a fallback from a
failed fetch (BUG 1), or a genuine source revision.
