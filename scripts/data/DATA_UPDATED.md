# Pulling and updating the data — Updated

Corrected/expanded version of `DATA.md`, reflecting what was learned validating
the pull (reproducing the 2025 vintage from the 2024 workbook). Covers running
the Python pull, getting its output into the MATLAB model's `indata/`, extending
the sample by a year, and — importantly — **verifying a pull actually worked**.
See `INSTRUCTIONS_FOR_AI.md` (repo root) for the MATLAB side, and
**[PIPELINE_BUGS.md](PIPELINE_BUGS.md)** for known pipeline bugs.

## 1. What the master pull does

`scripts/data/master-pull.py` rebuilds the two workbooks the MATLAB models read
(`DataInflShortLongConsUpdated.xlsx` and its no-consumption sibling
`DataInflShortLongUpdated.xlsx`) by starting from the existing workbook in
`indata/` and refreshing only the most recent years, country by country, from
each series' original source. Layout is flat: `master-pull.py`, `api_keys.py`,
and these docs all live in `scripts/data/`, reading/writing straight against
`indata/` (the `INDATA` constant, `HERE/../../indata`).

**Splice rule:** for every column, years **≤ `ANCHOR` (2020)** are kept exactly
as in the source workbook; years **`FETCH_START`–`FETCH_END` (2021–2025)** are
re-fetched. `cpi_*`/`rconpc_*` are index series (rebased); if the fresh source
level at the anchor year differs from the workbook by more than
`CHAIN_REANCHOR_PCT` (0.5%), the anchor year is pinned to the existing value and
new years chained forward by source growth rates. Rate columns (`stir_*`,
`ltir_*`, `baa_usa`) are placed directly.

Outputs, written into `indata/`:
- `DataInflShortLongConsUpdated_2025.xlsx` — full workbook (with `rconpc_*`)
- `DataInflShortLongUpdated_2025.xlsx` — same data, `rconpc_*` stripped

> **A failed or empty fetch is silent.** It is logged (`[FAIL ]`/`[warn ]`) and
> that column keeps its existing values while the workbook still saves. You will
> not get a non-zero exit — you must run the completeness check (§1c) to catch it.

### 1a. Requirements

```bash
pip install pandas openpyxl requests playwright
playwright install chromium
```
- `pandas`, `openpyxl`, `requests` — workbooks + HTTP fetches (FRED, OECD, IMF,
  World Bank, Bank of Canada, SNB, Norges Bank, Banque de France, …).
- `playwright` (+ Chromium) — **only** for the UK long rate (`ltir_uk_boe`); the
  Bank of England site needs a headless browser. On Linux this installs and
  launches fine.
  > **Bug (now fixed), see [PIPELINE_BUGS.md](PIPELINE_BUGS.md) BUG 1:** the BoE
  > fetch used to call a **Windows-only** asyncio policy unconditionally, which
  > threw `AttributeError` on Linux and silently dropped `ltir_uk`. It is now
  > guarded by `sys.platform`. If you are on an old checkout and see
  > `AttributeError: ... 'WindowsProactorEventLoopPolicy'` in a thread traceback,
  > apply that guard.
- `scripts/data/api_keys.py` — local, gitignored file defining `FRED_KEY` and
  `BDF_KEY`. Recreate it on a fresh checkout before running.

### 1b. Australia short rate — MANUAL step

`stir_au` is **not** fetched automatically — `sr_australia()` just reads
`indata/raw/f1.1-data.csv`. If that file is stale, `stir_au` silently freezes at
whatever year it contains while everything else refreshes. **Before a pull you
intend to use**, download the latest RBA **Table F1.1 — Interest Rates and
Yields, Money Market** CSV from <https://www.rba.gov.au/statistics/tables/> and
save it over `indata/raw/f1.1-data.csv` (same format: header block skipped via
`skiprows=11`, then `date,value` rows).

> If you (Claude) are running this and cannot fetch that file yourself, tell the
> user explicitly that `indata/raw/f1.1-data.csv` must be refreshed by hand from
> that RBA link first — do not pull silently with a stale Australia short rate.

### 1c. Running it + the completeness check

```bash
cd scripts/data
python master-pull.py
```
OECD endpoints are rate-limited, so OECD measures are fetched once for all
countries and cached under `scripts/data/_cache/`. `(... loaded ... from disk
cache)` on a re-run is expected; delete `_cache/` to force a fresh OECD pull.

**After every pull, verify the trailing years before using the output.** Sources
lag (World Bank `rconpc_*` and some OECD rates typically by ~a year) and the
script still saves a partially-blank last year. Open the output workbook(s) and
list any column that is blank for the last 1–2 years, e.g.:
```python
import openpyxl
ws = openpyxl.load_workbook("indata/DataInflShortLongConsUpdated_2025.xlsx", data_only=True)["Data"]
hdr = [ws.cell(1,c).value for c in range(1, ws.max_column+1)]
for r in range(ws.max_row-1, ws.max_row+1):
    yr = ws.cell(r,1).value
    blanks = [hdr[c-1] for c in range(2, ws.max_column+1) if ws.cell(r,c).value is None]
    print(yr, "missing:", blanks)
```
If anything is blank for the most recent year(s): tell the user which year/columns;
flag `stir_au` specially (→ the manual RBA step, not a normal lag); and state the
**last fully-complete year** (the year before the first incomplete one) — that,
not `FETCH_END`, is the real "last full year" the estimation should use for `T1`.

### 1d. Verifying the pull reproduces (regression harness)

To check the pipeline itself works, reproduce a known vintage: copy
`master-pull.py`, point its `SRC_XLSX` at the **previous** year's workbook and its
`OUT_XLSX`/`OUT_XLSX_NO_RCONPC` at a scratch dir (e.g. `indata_check/`), run it,
and diff cell-by-cell against the committed vintage. Split results into the
**≤2020 frozen block** (must be byte-identical — copied from the source) and the
**2021–FETCH_END block** (should match up to live source revisions). A ≤2020 diff
is a real bug; a 2021+ diff is a bug, a silent fallback (PIPELINE_BUGS BUG 1), or
a genuine provider revision. Exact reproduction is only expected shortly after the
original pull. This is exactly how BUG 1 was found.

## 2. Getting the pulled data into the MATLAB model

The MATLAB scripts read fixed filenames with **no** year suffix. All three models
plus `var01` read the **Cons** workbook `DataInflShortLongConsUpdated.xlsx` (74
columns; Model 1 uses it too and ignores `rconpc_*` — see
`INSTRUCTIONS_FOR_AI.md`). After a pull, back up the current plain file and copy
the new vintage over it:
```bash
cd indata
cp DataInflShortLongConsUpdated.xlsx DataInflShortLongConsUpdated_pre<NEW>backup.xlsx
cp DataInflShortLongConsUpdated_<NEW>.xlsx DataInflShortLongConsUpdated.xlsx
```
Keep the column layout identical — the scripts index columns positionally.

## 3. Changing the ending year

Two independent knobs:

**a. How far the pull fetches** — in `master-pull.py`:
```python
FETCH_START = 2021
FETCH_END   = 2025
```
Bump `FETCH_END` to the last **complete** year (the year *before* the current one —
sources rarely have a full calendar year until well into the next). New-year rows
are appended automatically. Do **not** move `ANCHOR`/`CHAIN_REANCHOR_YEAR` (2020).
The true last complete year can still be earlier than `FETCH_END` if a series lags
(§1c).

**b. Where the MATLAB estimation stops** — each `MainModel*.m` has
`T1 = find(Year==...)`. After getting the new data into `indata/` (§2), update
`T1` in every `MainModel*.m` you run (read the current value out of the script;
do not assume). The `_ReR/_1950/_df50/_var02..50/_unrestr` appendix specs are
pinned to 2016 on purpose. Figure scripts have no end-year anchors except
`Data_MY.xlsx`, which ends 2016 and is handled (see `INSTRUCTIONS_FOR_AI.md` §5).
