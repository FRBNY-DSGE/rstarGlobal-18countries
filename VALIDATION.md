# Pipeline validation / pre-flight checklist

What to check to trust the pipeline before (and after) a new-year data update, the
commands to run each check, and what a **healthy** result looks like. This is a
reusable pre-flight — re-run it whenever the data or the pull/figure scripts change.

> **Last run: 2026-07-20** (live, with MDN), on the 2025 vintage. Every check below
> passed; the only "difference" anywhere was the one known `ltir_uk` BoE-headless
> cell. Results from that run are noted under each check.

The scratch files/dirs these checks create (`master-pull-dryrun.py`, `indata/_dryrun/`,
`downloads_incoming/`, `key_rehearsal/`, OECD `_cache/`) are **throwaway — delete them
when done.** None should be committed.

---

## 1. Data pull reproduces a known vintage

**Why:** proves `master-pull.py` rebuilds the workbook correctly — the ≤`ANCHOR`
(2020) block must be copied verbatim from the source, and 2021→end re-fetched from
live sources.

**How** (reproduce last year's committed vintage from the year-before workbook):
```bash
cd scripts/data
cp master-pull.py master-pull-dryrun.py
# in the copy: SRC_XLSX -> DataInflShortLongConsUpdated_<Y-1>.xlsx,
#              OUT_XLSX / OUT_XLSX_NO_RCONPC -> ../indata/_dryrun/… , FETCH_END = <Y>
python master-pull-dryrun.py
```
Then diff the scratch output against the committed `…_<Y>.xlsx`, split into the
≤2020 frozen block and the 2021–<Y> fetched block (openpyxl cell-by-cell; see
`scripts/data/DATA_UPDATED.md` §1d for the method).

**Healthy result:**
- **≤2020 frozen block: 0 differing cells** (must be exact — any diff is a real bug).
- **2021–<Y> fetched block:** differences only where a live source legitimately
  revised, or the known `ltir_uk` gap (below).

**2026-07-20:** columns identical (74), ≤2020 = **0 diffs**, 2021–2025 = **1 diff**
(`ltir_uk` 2025 blank). Everything else matched to the last decimal.

---

## 2. API keys are valid (live check)

**Why:** the pull dies immediately without working `FRED_KEY`/`BDF_KEY`. Keys are
long-lived, so this rarely fails — but check before a pull; only chase a new key if
one is rejected (see `DATA_UPDATED.md` §1a for how to obtain each).

**How** (prints pass/fail only, never the key value):
```bash
cd scripts/data
python - <<'PY'
import requests, urllib.parse
from api_keys import FRED_KEY, BDF_KEY
r = requests.get("https://api.stlouisfed.org/fred/series",
    params={"series_id":"GNPCA","api_key":FRED_KEY,"file_type":"json"}, timeout=25)
print("FRED_KEY:", "VALID" if r.status_code==200 else f"BAD {r.status_code}")
url = ("https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/json?"
    + urllib.parse.urlencode({"order_by":"time_period_start",
        "refine":'series_key:"ECOFI.INR.FR.FITB_PA._Z.D"',
        "where":'time_period_start >= "2024-01-01"'}))
r = requests.get(url, headers={"Authorization": f"Apikey {BDF_KEY}", "Accept":"application/json"}, timeout=30)
print("BDF_KEY :", "VALID" if r.status_code==200 else f"BAD {r.status_code}")
PY
```
**Healthy result:** both `VALID`.
**2026-07-20:** both `VALID` (FRED returned series data; BdF returned 186 obs).

---

## 3. Australia auto-fetch + fallback

**Why:** `stir_au` now comes from the RBA F1.1 CSV fetched directly (no manual step);
the local `indata/raw/f1.1-data.csv` is a fallback. Confirm both paths agree.

**How:**
```bash
cd scripts/data
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("mp","master-pull.py")
mp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mp)  # main-guarded
auto = mp.sr_australia()                                   # live fetch
mp.RBA_URL = "https://www.rba.gov.au/statistics/tables/csv/NOPE.csv"      # force fallback
fb = mp.sr_australia()                                     # -> warns, uses local file
import numpy as np
print("auto vs fallback 2020-2025 match:",
      np.allclose(auto.loc[2020:2025], fb.loc[2020:2025], equal_nan=True))
PY
```
**Healthy result:** the fallback logs `[warn ] RBA F1.1 direct fetch failed …` and
the two series **match**.
**2026-07-20:** matched exactly (2020–2025); fallback warned and used the local file.

---

## 4. Figure scripts parse clean

**Why:** catch syntax breakage in any `MainModel*_MakeFigures.m` before a multi-hour,
80–110 GB render job.
```bash
cd scripts
matlab -nodisplay -r "for f={'MainModel1_MakeFigures.m','MainModel2_MakeFigures.m','MainModel3_MakeFigures.m'}; disp([f{1} ': '  ...]); checkcode(f{1}); end; exit"
```
**Healthy result:** no parse errors.
**2026-07-20:** M2 (edited this session) and M3 both parse clean; M1 additionally
**ran** (`make_figs_m1` → `FIGURES_DONE`, and regenerated the `update/` artifacts).

---

## Known accepted gap (not a failure)

- **`ltir_uk` (UK long rate) blank for the latest year on the cluster.** The Bank of
  England fetch can't run headless on the Linux nodes (fails safe). In a real update,
  fill that one cell from a machine where the BoE fetch works. See `PIPELINE_BUGS.md`.

## What each check would have caught
- **§1** — a mis-indexed column, a broken splice/re-anchor, or a source silently
  returning nothing (frozen-block drift or a whole column going blank).
- **§2** — an expired/missing key before it wastes a pull.
- **§3** — the RBA endpoint moving, or the fallback path being broken.
- **§4** — an edit to a figure script that only errors at run time.
