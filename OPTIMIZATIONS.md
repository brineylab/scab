# scab Codebase Technical Optimization Review

## Scope and Method
- Reviewed all Python source under `scab/`, CLI entrypoints in `bin/`, packaging metadata, docs in `docs/source/`, and tests in `scab/tests/`.
- Performed static code analysis (including `python -m compileall scab`) and manual API/doc consistency checks.
- Runtime execution was limited by missing optional scientific dependencies in this environment (`anndata`, `scanpy`, etc.), so findings are based on source-level analysis and reproducible control-flow inspection.

---

## 1) Duplicated / Redundant Code

### D1. Duplicate executable scripts
- **Where:** `bin/scabranger:1-30`, `bin/batch_cellranger:1-30`
- **Issue:** Files are effectively identical wrappers around `scab.cellranger.scabranger`.
- **Impact:** Duplicate maintenance surface; future fixes can drift.
- **Proposed fix:** Keep one script, remove the other (or make one a symlink/thin alias in packaging).

### D2. Duplicated pipeline entrypoint logic
- **Where:** `scab/cellranger/scabranger.py:1325-1385` and `1387-1444`
- **Issue:** `cellranger_pipeline()` and `main()` duplicate nearly all orchestration logic.
- **Impact:** Bug fixes must be duplicated; drift risk is high.
- **Proposed fix:** Extract a single private runner (e.g. `_run_pipeline(project_dir, config_file, debug)`), call from both wrappers.

### D3. Two clonify implementations that have diverged
- **Where:** `scab/vdj.py:478-657`, `scab/tools/clonify.py:50-308`
- **Issue:** Same conceptual algorithm exists in two modules with different behavior and different bug states.
- **Impact:** API confusion and inconsistent lineage assignments.
- **Proposed fix:** Choose one canonical implementation, deprecate/remove the other, and add compatibility shim if needed.

### D4. Duplicate pairwise distance implementations
- **Where:** `scab/vdj.py:660-670`, `scab/tools/clonify.py:311-355`
- **Issue:** `_clonify_distance()` and `pairwise_distance()` are equivalent logic duplicated in separate modules.
- **Impact:** Easy to introduce algorithm drift.
- **Proposed fix:** Centralize in one shared utility function and import from both call sites.

### D5. Alias functions with duplicated signatures/docs
- **Where:** `scab/tools/cellhashes.py:50-190` and `298-395`; `scab/io.py:393-506`
- **Issue:** `demultiplex()`/`assign_cellhashes()` and `read()`/`load()`, `write()`/`save()` carry near-duplicate doc and parameter surface.
- **Impact:** Documentation and behavior drift over time.
- **Proposed fix:** Keep alias wrappers minimal and short; avoid full duplicated docblocks.

### D6. Large volumes of dead commented-out legacy code
- **Where:** `scab/tl.py` (1144 lines, ~886 commented), `scab/pl.py` (3280 lines, ~1663 commented)
- **Issue:** Historical implementations are retained inline as comments.
- **Impact:** Obscures active behavior and slows maintenance.
- **Proposed fix:** Remove commented legacy code from runtime modules; keep historical context in git history or migration docs.

---

## 2) Structural / Design Maintainability Hotspots

### S1. `scab/tools/clonify.py` contains hard runtime defects
- **Where:** `scab/tools/clonify.py:165`, `290`, `307`
- **Issue:**
  - `np.random_choice` (invalid NumPy API; should be `np.random.choice`)
  - `for s, l in assignment_dict.values()` iterates values only
  - `lineage_sizes` assigned but never defined
- **Impact:** Function is currently non-functional in common paths.
- **Proposed fix:** Repair implementation, add focused unit tests for small and multi-heavy cases, or remove in favor of `scab.vdj.clonify`.

### S2. `scab/plots/heatmap.py` is incomplete/broken
- **Where:** `scab/plots/heatmap.py:371-422`
- **Issue:**
  - Calls `get_adata_values(..., receptor_type=..., chain_type=...)` with unsupported kwargs
  - References undefined `agg_method`
  - File ends without rendering/return path
- **Impact:** Function is unusable; dead API surface.
- **Proposed fix:** Either complete implementation and add tests/docs, or remove from codebase until ready.

### S3. `classify_specificity()` has multiple correctness bugs
- **Where:** `scab/tools/specificity.py:155-164`, `187-200`, `210-236`
- **Issue:**
  - Auto-detected `agbcs` becomes boolean mask, not column names (`159-161`)
  - Mutates `groups` during iteration (`200`)
  - If `threshold_dict` supplies threshold, `adata_groups[group_name]` may be missing (`210-227`)
  - `raw_bc_thresholds` used even when not defined (`234-235`)
- **Impact:** High risk of runtime errors and invalid specificity calls.
- **Proposed fix:** Refactor loop to immutable iteration + validated intermediate structures; add tests for all threshold modes.

### S4. `combat()` computes UMAP on the wrong object
- **Where:** `scab/tools/batch_correction.py:109-113`
- **Issue:** UMAP is run on `adata`, but function returns `adata_combat`.
- **Impact:** Post-correction embedding options can silently do nothing useful.
- **Proposed fix:** Run `umap()` on `adata_combat` and return that object consistently.

### S5. Sample-specific reference resolution bug in Cell Ranger config parsing
- **Where:** `scab/cellranger/scabranger.py:38`, `227-233`
- **Issue:** `Config._parse_config_file()` uses `name` (from `unicodedata.name`) instead of `sample_name` when selecting references.
- **Impact:** Per-sample references likely never resolve correctly; behavior silently falls back.
- **Proposed fix:** Replace `name` with `sample_name`; remove `from unicodedata import name`.

### S6. Broken index CSV filename
- **Where:** `scab/cellranger/scabranger.py:815`
- **Issue:** Looks for `index_seqences.csv` but repository file is `index_sequences.csv`.
- **Impact:** Element `bases2fastq` manifest path can fail at runtime.
- **Proposed fix:** Correct filename and add regression test around `_load_index_sequences()`.

### S7. `Config.__repr__` references non-existent attributes
- **Where:** `scab/cellranger/scabranger.py:146-154`
- **Issue:** Uses `self.reference` and `self.transcriptome` which are not defined.
- **Impact:** Debug/inspection methods can throw exceptions.
- **Proposed fix:** Update to `gex_reference`/`vdj_reference` consistently or remove brittle repr detail.

### S8. Incorrect sample gating before `cellranger multi`
- **Where:** `scab/cellranger/scabranger.py:1368-1370`, `1428-1430`
- **Issue:** Condition checks `if not sample.libraries`, but `sample.libraries` is config-defined and often truthy even when no FASTQ paths exist.
- **Impact:** Pipeline may run `cellranger multi` with empty/invalid library paths.
- **Proposed fix:** Gate on available FASTQ paths (`any(lib.fastq_paths for lib in sample.libraries)`).

### S9. Heavy shell-command orchestration with `shell=True`
- **Where:** `scab/cellranger/scabranger.py:549`, `622`, `709`, `757`, `992` (and related command builders)
- **Issue:** String-built shell commands with user-provided paths/options.
- **Impact:** Injection risk, quoting fragility, and platform inconsistency.
- **Proposed fix:** Use `subprocess.run([...], check=True)` with argument lists; capture and structure stdout/stderr safely.

### S10. Library code uses process exits and print-side effects
- **Where:** e.g. `scab/vdj.py:571`, `888-889`, `915-916`; `scab/tools/specificity.py:171-174`; `scab/cellranger/scabranger.py:753-756`
- **Issue:** `sys.exit()` and direct `print()` are used in library functions.
- **Impact:** Hard to integrate into larger applications and hard to test.
- **Proposed fix:** Raise typed exceptions and let CLI layer format user-facing output.

### S11. Monolithic modules as hotspot risks
- **Where:** `scab/pl.py`, `scab/tl.py`, `scab/cellranger/scabranger.py`, `scab/vdj.py`, `scab/ont/vdj.py`
- **Issue:** Large files with mixed responsibilities (I/O, business logic, plotting, CLI, orchestration).
- **Impact:** High cognitive load and slow safe refactoring.
- **Proposed fix:** Split by concern (API facade vs algorithms vs orchestration vs rendering).

### S12. Wildcard exports obscure API boundaries
- **Where:** `scab/tl.py:36-39`
- **Issue:** `from .tools.<module> import *`
- **Impact:** Hidden namespace coupling and accidental API changes.
- **Proposed fix:** Explicit imports and explicit `__all__`.

### S13. Mutable default argument in consensus pipeline
- **Where:** `scab/ont/consensus.py:28`
- **Issue:** `alignment_kwargs={"guide_tree": "upgma"}` is mutable default and then mutated (`120`).
- **Impact:** Cross-call state bleed and non-obvious behavior.
- **Proposed fix:** Default to `None`, initialize per call.

### S14. Potential unbound return variable in consensus worker
- **Where:** `scab/ont/consensus.py:99-167`
- **Issue:** `consensuses` may be referenced at return after exception before assignment.
- **Impact:** Secondary failure masks root cause.
- **Proposed fix:** Initialize `consensuses = []` before `try`.

---

## 3) Documentation / Docstring Drift

### DOC1. `read_10x_mtx()` doc uses wrong parameter names
- **Where:** `scab/io.py:160`, `169` vs signature at `48-91`
- **Issue:** Doc mentions `ignore_cellhash_regex_case` / `ignore_agbc_regex_case`, code uses `ignore_cellhash_case` / `ignore_agbc_case`.
- **Proposed fix:** Align docstring with actual API names.

### DOC2. `merge_bcr()` / `merge_tcr()` doc advertises unsupported formats/params
- **Where:** `scab/vdj.py:216-243`, `321-347`
- **Issue:** Doc references `delimited`, `json`, delimiter params, and `abstar_output_format` that are not active in signature.
- **Proposed fix:** Remove legacy parameters from docs or reintroduce support intentionally.

### DOC3. Wrong return location in `merge_tcr()` doc
- **Where:** `scab/vdj.py:359-362`
- **Issue:** Says TCR is at `adata.obs.bcr`; code uses `adata.obs.tcr`.
- **Proposed fix:** Correct doc text.

### DOC4. Examples page contains invalid/outdated API calls
- **Where:** `docs/source/examples.rst:110`, `123`, `126`
- **Issue:**
  - `adata = adata.tl.demultiplex(...)` (incorrect namespace)
  - `scab.pl.umap(..., colors=[...])` (doesn’t match current wrapper signature)
  - `scab.vdj.group_clonotypes(...)` (function does not exist)
- **Proposed fix:** Execute examples in CI/doctest or notebooks to prevent drift.

### DOC5. Python version drift between README and package metadata
- **Where:** `README.md:30` vs `setup.py:61`
- **Issue:** README says Python 3.6+, package requires >=3.8.
- **Proposed fix:** Update README and docs to 3.8+ (or adjust package requirement).

### DOC6. scabranger docs/config mention unsupported config blocks
- **Where:** `scab/cellranger/example.yaml:48-51`, `124-127`; parser in `scab/cellranger/scabranger.py:210-247`
- **Issue:** Example includes `fastqs` and `compress`, parser does not consume these keys.
- **Proposed fix:** Either implement those blocks or remove them from docs/examples.

### DOC7. API section text mismatches content
- **Where:** `docs/source/api.rst:76`
- **Issue:** Tools section description repeats preprocessing language (“Filtering and normalization...”).
- **Proposed fix:** Update section narrative to reflect actual tools (`batch_correction`, `cellhashes`, etc).

---

## 4) Other General Code Smells

### C1. `io.read_10x_mtx()` has unused/overwritten params
- **Where:** signature `scab/io.py:71,74`; overwritten at `312`, `315`
- **Issue:** `hashes` and `agbcs` parameters are accepted but ignored/overwritten.
- **Impact:** Misleading API and user confusion.
- **Proposed fix:** Implement support or remove parameters.

### C2. `io.write()` is not robust to `pathlib.Path`
- **Where:** `scab/io.py:448-470`
- **Issue:** Uses `.endswith()` on `h5ad_file` without coercion to string.
- **Impact:** Fails with valid `Path` inputs.
- **Proposed fix:** Normalize with `pathlib.Path` and use suffix checks.

### C3. Broad exception handling that can hide root failures
- **Where:** e.g. `scab/tools/cellhashes.py:152-167`, `scab/ont/barcodes.py:100-103,130-132`, `scab/ont/consensus.py:147-160`
- **Issue:** Catches broad exceptions and continues.
- **Impact:** Silent partial failure and hard debugging.
- **Proposed fix:** Catch specific exceptions; accumulate structured error reports; fail fast in strict mode.

### C4. Debug print leftovers in production flow
- **Where:** `scab/cellranger/scabranger.py:584-585`, `592-593`
- **Issue:** Unconditional `print()` in `mkfastq()`.
- **Impact:** Noisy logs and inconsistent output routing.
- **Proposed fix:** Replace with logger calls under debug level.

### C5. Syntax warnings from invalid escape sequences
- **Where:** `compileall` warnings in plotting modules and splash strings
- **Issue:** Strings like `"\mathregular"` or ASCII art backslashes trigger warnings.
- **Impact:** Noise and potential future strict-mode failures.
- **Proposed fix:** Use raw strings (`r"..."`) or escape backslashes.

### C6. Test coverage misses high-risk modules
- **Where:** `scab/tests/` (no tests for `scab/cellranger`, `scab/tools/specificity`, `scab/tools/clonify`, `scab/plots/*`)
- **Issue:** Critical modules are untested; some existing tests are commented out (`test_batch_correction.py:37-72`).
- **Impact:** Regressions likely in production-only code paths.
- **Proposed fix:** Add targeted unit/integration tests for broken/high-risk paths; re-enable gated tests with optional markers.

### C7. Suspicious copy/paste in TCR test fixtures
- **Where:** `scab/tests/test_io.py:46-58`
- **Issue:** TCR fixtures point to `bcr` file paths.
- **Impact:** Reduced confidence in TCR-specific code paths.
- **Proposed fix:** Use real TCR fixture files and add TCR merge assertions.

---

## Recommended Remediation Order

1. **Fix correctness blockers first:** `tools/clonify.py`, `plots/heatmap.py`, `tools/specificity.py`, `batch_correction.combat()`, `cellranger Config` sample-reference bug, index filename typo.
2. **Stabilize architecture:** remove duplicate pipeline/script paths, split monolith files, replace wildcard imports, remove dead commented code.
3. **Repair docs/examples:** bring docs into executable parity with current public API.
4. **Harden quality gates:** add tests for currently uncovered modules and enforce smoke checks in CI (import + basic call path).

