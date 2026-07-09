# Fixes applied for the 18-country / 2025-vintage run

This documents every change made to get the repository to estimate all three
models (plus the `var01` alternative-prior spec) on **18 countries** with data
through **2025**, and to produce the full figure set. As shipped, **none of the
three models could run** — the issues below were pre-existing (data files not
wired, parse errors, half-wired Denmark), not artifacts of the data update.

All script/data edits are on `master` and were uncommitted at time of writing.
Backups made: `indata/DataInflShortLongConsUpdated_pre2025backup.xlsx` (the old
plain workbook that ended 2020) and `scripts/MainModel1_var01.m.stale_backup`
(the old incompatible var01).

---

## 1. Data wiring

- **Created the output directories** `results/` and `results/18/`. They did not
  exist, so `save(...)` / `copyfile(...)` would have crashed at the end of a run.
- **Wired the 2025 workbook to the plain filename.** The scripts read fixed names
  with no year suffix, but the repo only shipped year-suffixed vintages plus a
  stale plain Cons file ending in 2020. Copied
  `DataInflShortLongConsUpdated_2025.xlsx` → `DataInflShortLongConsUpdated.xlsx`.
- **Pointed Model 1 at the Cons workbook.** `MainModel1.m` had `xlsread(... 'DataInflShortLongUpdated.xlsx')`
  (the non-cons name, which was missing entirely), but its `country_start=[0,4,8,…,69]`
  loop reads the **74-column Cons layout** (4 cols/country incl. `rconpc`, skipping
  the one-off `baa_usa` column). The 56-column non-cons file would silently
  mis-index every country from `au` onward. Changed line 13 to read
  `DataInflShortLongConsUpdated.xlsx` (Model 1 just ignores the `rconpc` columns).

## 2. Sample period

- Bumped `T1 = find(Year==2024)` → `find(Year==2025)` in `MainModel1.m`,
  `MainModel2.m`, `MainModel3.m` (and inherited by `MainModel1_var01.m`). The dead
  `T0=100; T1=144;` lines earlier in M2/M3 were left alone (they are overwritten).

## 3. Estimation-script parse errors

- **Unclosed `codes = {…}` cell brace** in `MainModel1.m` (line ~79) and
  `MainModel2.m` (line ~99): the list ended with `;` instead of `};`, so MATLAB
  swallowed the next line (`Nc = numel(codes)`) and threw
  *"Incorrect use of '=' operator"*. Added the closing brace. `MainModel3.m` was
  already correct.

## 4. Denmark wiring (the "18countries" fix)

- **Model 1 — Denmark half-wired.** `codes` included `dk` (so `Nc=18`) and
  `Cadd2/Cadd3/Psi/SC0tr/S0tr/P0tr` were all sized for 18, but `Y`, `Mnem`, and
  `Ctr` only listed **17** countries (through `pt`). This overflowed at the
  moment-print `Y(:,Nc*3)` (needs 54 columns, had 51). Added the three `dk` rows
  to `Y`, `Mnem`, and `Ctr` (loadings `Stir→1 1 0`, `Infl→0 1 0`, `Ltir→1 1 1`).
  Result: n=54, r=57.
- **Model 2 — estimated for only 17 countries.** Its `codes` list stopped at `pt`.
  Added `dk` to `codes` and the three `Y`/`Mnem`/`Ctr` blocks (Model-2 loadings are
  4-wide, `Stir→1 1 0 -1`, `Infl→0 1 0 0`, `Ltir→1 1 1 -1`; the trailing `Baa_us`
  observation stays last). Everything else in M2 is written in terms of `Nc`, so the
  priors/initial conditions auto-scaled to 18 (n=55, r=58). No prior editing needed.
  *(The `Country` uppercase display array is a dead variable — cosmetic only.)*

## 5. `var01` (disperse-prior spec, for Model 1 Figure 2)

- The shipped `MainModel1_var01.m` was a **stale 17-country copy** using the old
  **67-column "irregular" layout** (3 cols for the first 7 countries, 4 for the
  rest) via hard-coded `X(:,col)` indexing. No current data file has that layout,
  so it mis-indexed against everything.
- **Regenerated it from the fixed `MainModel1.m`**, changing only three lines:
  the header comment, `filename` → `OutputModel1_var01.mat`, and the one prior
  knob — the `SC0tr` divisor **`/100` → `/1`**. That divisor *is* the whole
  var-family: `var01=/1`, `var02=/2`, `var05=/5`, `var10=/10`, `var25=/25`,
  `var50=/50`, baseline `MainModel1=/100`. It sets the prior variance of the
  innovations to the trends — smaller divisor ⇒ more disperse prior ⇒ trends free
  to move more. Nothing else differs from the baseline.

## 6. Figure-script fixes

- **`MainModel3_MakeFigures.m` — `codes` 17 → 18.** The figure derives `Nc` from
  its own `codes` and uses it as the **stride** to slice per-country idiosyncratic
  trends out of the state (`base=6`; `idx_pi=base+Nc+(1:Nc)`, etc.). With the
  estimate at 18 but the figure at 17, `idx_pi` onward read the wrong offsets — a
  **silent mis-indexing** that corrupts every per-country trend. Synced to 18.
- **`MainModel1_MakeFigures.m` — `country_colors` 17 → 18 rows.** The Figure-3a
  country-overlay block used 18-country `codes` but a 17-row color array, so the
  `k=1:18` loop went out of bounds at Denmark. Added an 18th color row (applied to
  all three color arrays; harmless for the blocks that only loop to 17).
- **`MainModel1_MakeFigures.m` — Figure 2 guarded.** Wrapped the `var01`-dependent
  block in `if exist('../results/OutputModel1_var01.mat','file')` so the rest of
  the M1 figures still generate even if `var01` hasn't been estimated.
- **`MainModel1_MakeFigures.m` — Figure 5 (MY regression) length fix.** ← *the
  latest fix.* `Data_MY.xlsx` (the demographic "MY" regressor) ends in **2016**
  (147 rows), but the estimation sample now runs to **2025** (`T=156`), so
  `[ones(T,1) MY_us]` failed to concatenate. Replaced the 8 `fit_*_MY` lines with
  helpers `nMY = size(MY,1)`, `padMY = @(v)[v(:);nan(T-nMY,1)]`,
  `fitMY = @(q,x)[ones(T,1) padMY(x)]*regress(q(1:nMY,3),[ones(nMY,1) x(:)])` —
  i.e. run the regression on the overlapping years (1870–2016) and NaN-pad the
  fitted line past 2016 (it simply stops there). This unblocked Figure 5 **and**
  appendix figures A1 and A2, which run after it.

  *Note:* the other figure scripts have **no hard-coded end-year anchors** — they
  read the full `Year` vector, so new years appear automatically. `Data_MY.xlsx`
  was the sole exception because it is external demographic data that ends in 2016.

## 7. Batch infrastructure created

The batch drivers referenced in the old instructions pointed at a *different*
repo path and did not exist here. Created:

- `scripts/run_model1.m`, `run_model2.m`, `run_model3.m`, `run_model1_var01.m` —
  each `cd`s into `scripts/`, runs the estimation, copies M1/M2 output into
  `results/18/`, and `exit`s.
- `scripts/make_figs_m1.m`, `make_figs_m2.m`, `make_figs_m3.m` — each `cd`s in,
  runs the matching `*_MakeFigures.m` inside a `try/catch` (so a mid-script error
  is logged and MATLAB still `exit`s), prints `FIGURES_DONE model N` on success.

## Result

All four estimates on disk (`OutputModel1.mat`, `OutputModel1_var01.mat`,
`OutputModel2.mat`, `OutputModel3_new.mat`) and the full figure set generated
(20 main + 90 appendix PDFs in `figures/` and `figures/appendix/`), all three
figure scripts reporting `FIGURES_DONE`.
