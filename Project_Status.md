# Project status

Snapshot of the work done to get the Global r\* (18-country) model estimating and
plotting on data through **2025**, and to validate the data-pull pipeline.
Companion docs: **[Fixes.md](Fixes.md)** (estimation/figure fixes),
**[INSTRUCTIONS_UPDATED.md](INSTRUCTIONS_UPDATED.md)** (run guide, incl. 2025→2026),
**[scripts/data/DATA_UPDATED.md](scripts/data/DATA_UPDATED.md)** (data pull),
**[scripts/data/PIPELINE_BUGS.md](scripts/data/PIPELINE_BUGS.md)** (pull bugs).

_Last updated: 2026-07-09._

---

## 1. Headline

- **All four specifications estimated on the 18-country / 2025 sample** and their
  figures produced: Model 1 (Baseline), Model 1 `var01` (disperse prior), Model 2
  (Convenience Yield), Model 3 (Consumption).
- **Data-pull pipeline validated** by reproducing the 2025 vintage from the 2024
  workbook — it reproduces exactly except the UK long rate (a real, now-fixed
  Linux bug) and a sub-basis-point Banque-de-France revision.
- Two batches of fixes were required — the repo as shipped could not run any model,
  and the data pull could not fetch the UK long rate on Linux.

## 2. Estimation & figures

Run on the cluster via `matlab20a-batch-withemail`; drivers `scripts/run_model*.m`
and `scripts/make_figs_m*.m`. Outputs (NOT in git — 20–38 GB each) in `results/`
and `results/18/`; **20 main + 90 appendix PDFs** in `figures/` + `figures/appendix/`.

| Spec | Output `.mat` | Figures |
|---|---|---|
| Model 1 Baseline | `results/OutputModel1.mat` (+ `results/18/`) | Fig 1, 2, 3a/b, 4a/b, 5, A1, A2 |
| Model 1 `var01` | `results/OutputModel1_var01.mat` | feeds Model 1 Figure 2 |
| Model 2 Conv. Yield | `results/OutputModel2.mat` (+ `results/18/`) | Fig 6a/b/c, appendix |
| Model 3 Consumption | `results/OutputModel3_new.mat` | Fig 7a/b/c, A18 |

Sample: `T1 = find(Year==2025)`. All 18 countries incl. Denmark. See Fixes.md for
the full list of estimation/figure repairs (data wiring, `codes{}` braces, the
Denmark wiring in Models 1 & 2, `var01` regeneration, and figure-script fixes
including the `Data_MY`-ends-2016 / Figure 5 length fix).

**Tables.** `Tables.m` (companion to `makeTables.m`) produces the Global-vs-US
r\* summary tables — change in r-bar over 1990–2019 and 2019–2025 (median, 90%
interval, P(change<0)) for all three models, each decomposed: Model 2 into
r\*/−cy/other(m), Model 3 into r\*/g/β/−cy. It writes
`tables/GlobalUS_{Model1,Model2,Model3,Combined}.tex` (Combined = all three
panels) plus `tables/GlobalUS_Levels.tex` — the end-of-sample **level** of
global/US r-bar (the value at the right edge of Figure 1) for Models 1 & 2:
median, 90% interval, {P(r-bar<0)}. For 2025: Model 1 global r-bar median
**0.16**, US **0.38** (Model 2: 0.45 / 0.78). Run via driver `run_tables.m`
(~110 GB — loads `results/18/OutputModel{1,2}.mat` and the 37 GB
`results/OutputModel3_new.mat`). It was consolidated from an identical
`Tables.m`/`Tables_Elena.m` pair; every addition was verified to leave the other
tables byte-for-byte unchanged. Its `find(Year==2025)` anchors (one per model)
must be bumped on a new-year update. `makeTables.m` builds the appendix A1a/A1b
decomposition tables and is unchanged.

## 3. Data-pull validation (this session's check)

**Method:** copied `master-pull.py` to a throwaway script pointing `SRC_XLSX` at
`DataInflShortLongConsUpdated_2024.xlsx` and outputs at a scratch `indata_check/`,
ran it, and diffed cell-by-cell against the committed 2025 workbooks. (Scratch
dir + copy since deleted; method recorded in DATA_UPDATED.md §1d.)

**Result — reproduces faithfully:**

| Block | Cons workbook | Non-cons |
|---|---|---|
| ≤2020 (frozen) | 0 / 11,174 differ | 0 / 8,456 differ |
| 2021–2025 (re-fetched) | 6 / 370 differ | 6 / 280 differ |

The only differing columns:
- **`ltir_uk` 2025** — blank; the Bank of England fetch cannot run headless on the
  Linux nodes (BUG 1 + BUG 2, see PIPELINE_BUGS.md). Now fixed to fail gracefully;
  proper fix (sync-Playwright rewrite) still pending.
- **`ltir_fr` 2021–2025** — sub-basis-point differences (≤0.006 pp) from a tiny
  Banque-de-France webstat revision between the Jul 8 pull and the Jul 9 re-pull.
  Benign; the expected "live sources revise" behavior.

**Conclusion:** the data-update instructions work end-to-end. The ≤2020 anchor is
reconstructed exactly and 2021–2025 re-fetch correctly for every series except the
UK long rate.

## 4. Bugs found & fixed this session

Estimation/figures (details in Fixes.md):
1. Output dirs `results/`, `results/18/` missing → created.
2. Data not wired to plain filenames; Model 1 needed the 74-col Cons layout.
3. `T1` 2024 → 2025 across scripts.
4. Unclosed `codes = {…}` brace in Model 1 & Model 2.
5. Denmark half-wired in Model 1; Model 2 was only 17 countries → both wired to 18.
6. `var01` regenerated from the fixed baseline (old 67-col layout was incompatible);
   sole substantive difference is the `SC0tr` divisor (100 → 1).
7. Figure scripts: Model 3 `codes` 17→18 (mis-indexing), Model 1 `country_colors`
   18th row, Figure 2 var01 guard, Figure 5 MY-regression length fix.

Consumption model (Model 3) — **observation-matrix bug** (found 2026-07-14 while
investigating why Model 3's r-bar decomposition looked off vs Models 1 & 2):
- In `MainModel3.m` the observation rows are `Stir, Infl, Ltir, Dcons, Baa`, so the
  single `Baa_us` row is the **last** row = `4*Nc+1` (73 for Nc=18). The original
  7-country code (FRBNY repo) correctly set the Baa idiosyncratic loadings there
  (`Cadd2(29,1)=1; Cadd3(29,1)=1`, 29 = 4·7+1). The 18-country port instead wrote
  **`3*Nc+1`** (=55 = `Dcons_us`), apparently copied from Model 2 (which has no
  `Dcons` group, so its Baa sits at `3*Nc+1`). Effect: `Dcons_us` spuriously loaded
  on the US idio inflation & term-spread trends, and `Baa_us` was missing those
  loadings — corrupting the US Baa/consumption equations and, plausibly, inflating
  Model 3's convenience-yield decline (−2.54 vs Model 2's −1.53).
- Everything else (priors `SC0tr/S0tr/P0tr/Psi/df0tr`, world-trend `Ctr` loadings,
  `Cadd4` consumption block) matches the original exactly. One non-bug prior
  difference: Model 3's inflation trend prior is `2` (var 4) vs `√2` (var 2) in
  Models 1 & 2 — but that value is inherited from the original consumption code.
- **Fix:** `Cadd2/Cadd3(3*Nc+1,1)` → `(4*Nc+1,1)` (and the harmless `Cadd1` line).
  Re-estimated Model 3 in two versions: **A** = fix + original inflation prior `2`
  (`MainModel3.m` → `OutputModel3_new.mat`); **B** = fix + Models-1&2 inflation
  prior `√2` (`MainModel3_B.m` → `OutputModel3_B.mat`). Pre-fix output preserved as
  `results/OutputModel3_buggy.mat`.
- **Outcome (2026-07-15):** the fix pulled the **US −cy** decline into line with
  Model 2 — 1990–2019 US −cy went from −2.01 (buggy) to **−0.97** (A) / **−0.86**
  (B), vs Model 2's −0.80. A and B are nearly identical, i.e. the inflation-trend
  prior barely affects the r\* decomposition. Global r-bar decline stays steeper
  than Models 1 & 2 (≈ −5.2 vs −3.3 / −3.6 over 1990–2019). Full results for both
  variants: tables `tables/GlobalUS_Model3.tex` (= A) and `GlobalUS_Model3_B.tex`;
  figures `figures/fig7a/b/c-Model3_*.pdf` (= A) and `figures/model3_B/fig7a/b/c…`
  (= B, via `MainModel3_B_MakeFigures.m` / `make_figs_m3_B.m`).

Data pipeline (details in PIPELINE_BUGS.md):
- **BUG 1** — `master-pull.py` called a Windows-only asyncio policy unconditionally,
  killing the BoE/UK fetch on Linux. Fixed (guard by `sys.platform`).
- **BUG 2** — with BUG 1 fixed, the BoE fetch (async Playwright in a worker thread,
  no join timeout) hangs forever on headless Linux. Mitigated (daemon thread +
  bounded 150s join → graceful `[FAIL]` fallback). Proper fix = sync-Playwright
  rewrite; until then `ltir_uk` must be refreshed on a non-headless machine.

## 5. Documentation produced

- `Fixes.md` — estimation/figure changelog.
- `INSTRUCTIONS_UPDATED.md` — run guide with a 2025→2026 checklist (original
  `INSTRUCTIONS.md` left untouched).
- `scripts/data/DATA_UPDATED.md` — corrected data-pull guide (completeness check,
  Australia manual step, UK/Playwright reality, reproduce-a-vintage method).
- `scripts/data/PIPELINE_BUGS.md` — BUG 1 & BUG 2.
- `Project_Status.md` — this file.

## 6. Git

Branch **`fix/18country-2025`** off `master` (master untouched, nothing pushed):
- `4e47506` — fixed scripts + batch drivers + Fixes.md + INSTRUCTIONS_UPDATED.md + `.gitignore`.
- `72c90e2` — 2025 workbook wired to plain name + `scripts/data/{DATA.md,INSTRUCTIONS.md}`.
- `ed1780e` — `master-pull.py` pipeline fixes + DATA_UPDATED.md + PIPELINE_BUGS.md + Project_Status.md.
- `71cacc8` — `Tables.m` (Global/US r\* tables) + `run_tables.m` + INSTRUCTIONS_UPDATED.md tables docs.
- `ecb2138` — Project_Status.md: tables entry + refreshed git log.
- `e5f05e4` — `Tables.m`: end-of-sample r-bar levels table (`GlobalUS_Levels.tex`).
- `c6ab73e` — `Tables.m`: Model 3 (Consumption) Global/US table + folded into Combined.
- `ce71af0` — `MainModel3.m` Cadd Baa-row bug fix (A) + `MainModel3_B.m` (aligned
  inflation prior) + `run_model3_B.m`.
- `3654b96` — `Tables.m`: emit `GlobalUS_Model3_B.tex`; canonical Model 3 table now
  reflects the corrected version A.
- (+ Project_Status.md / INSTRUCTIONS_UPDATED.md doc updates)

Excluded from git throughout: `scripts/data/api_keys.py` (secrets), `results/`
(20–38 GB `.mat`), SGE logs, and test scratch (`indata_check/`, `_cache/`).

## 7. Known open items

- **`ltir_uk` on the cluster** — headless BoE fetch still doesn't retrieve data
  (fails safe). Needs the sync-Playwright rewrite, or update that one series
  off-cluster. Currently `ltir_uk` 2025 in the shipped workbook came from a machine
  where the fetch worked.
- Regenerated **figure PDFs** and the root **`INSTRUCTIONS.md`** are intentionally
  left uncommitted (per prior scoping).
- 2026 data are not yet published; when they are, follow the 2025→2026 checklist
  in INSTRUCTIONS_UPDATED.md §0 and DATA_UPDATED.md.
