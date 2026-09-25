# Provenance

## Authoritative manuscript inputs

The uploaded `CODEX_BEHAVIORAL_COVERAGE_BOOTSTRAP.zip` supplies the current `ICLR2027_Behavioral_Coverage_TopFigures_Source.zip` and the separately supplied `references_cleaned.bib`. The paper ZIP SHA256 is `0290d450f0b22f50a030dfb0c5d851eb82927c5458393c65b62b327e6921d7e6`; the cleaned bibliography SHA256 is `6471551902ce9a2ec61cde6c93dfc687b7505b765374de1464726b1ad35e6ca0`. All 261 source-package manifest entries were verified before restructuring. The package PDF and the earlier separately attached PDF have different hashes; the user's later instruction makes the uploaded source package authoritative.

Original provenance, identity records, historical QA and revision notes are preserved under `manifests/paper_package/`. Those historical checks are evidence from the upload, not claims that the current build performed them. Current checks write to `build/qa/` and are summarized in `REPRO_STATUS.md`.

`manifests/input_sha256.tsv` maps every bundled frozen input or archived vector source file to its original archive/member, stage, size, hash, and paper role. `data_map.json` records the lossless path relocation used to recreate the original script layout. Original CSV/JSON/Parquet/NPZ bytes were not rounded, repaired, or rewritten. The vector scalar CSV is explicitly a frozen lossless derivative, not original model output; it is verified independently against the retained Parquet. Its nested weight columns remain in that Parquet.

The uploaded `build_assets.py`, `build_vector_evidence.py`, and `read_scalar_parquet.py` are copied without scientific edits. A wrapper runs them in a disposable directory containing only this repository's inputs. Generated CSVs and QA are separate from frozen inputs. The two inline manuscript tables were extracted into table inputs and regenerated from the original control JSON and checkpoint CSV; their rendered table text is checked against the source package's table fragments. Main prose, claims, standalone panel definitions, conference styles, and figure assembly are preserved.

## Modern LLMs

The read-only upstream roots are the `PIER_LLM_ECOSYSTEM_GOLDMINE_V2`, `...V2_1_REANALYSIS`, `...V2_2_TRACKWISE_REANALYSIS`, and `...V2_3_INTERFACE_VALIDATION` directories under `/work/Users/leiyo`. Historical `/work/Lei` aliases are recorded as provenance and are not required by paper replay.

All 24 paper-bundled LLM inputs listed in the original archive identity manifest were matched against the corresponding server ZIP members. The V2.3 final ZIP hash is `ba24123066a16aa29af60fa924589256fd7fe8566ecd9a2017f12dccaeb682a1` and matches its final completion marker. The vector Parquet hash is `1604bc3b71ba193fea699dec26ec24965230a73435ba69588e9f1bf82e1a05a9` and matches its successful stage-11 marker. `llm_paper_upstream_identity.json` contains the per-member comparisons.

Exact copied source/configuration files and their hashes are listed in `manifests/upstream_sources.tsv`; copied-component tree hashes are in `source_tree_hashes.json`. These non-Git experiment trees are not assigned invented commits. V2.1 is retained as a dependency of the later analyses, not promoted to a new manuscript experiment. V2.3's failed initial Stage-C attempts, authorized feasibility repair, recovery completion, and environment changes remain visible in the copied records.

The final source audit rehashed all 450 recorded source mappings against their original locations. `sha256`/`size_bytes` identify the original source; `maintained_sha256`/`maintained_size_bytes` identify the repository copy. The six differing mappings are the two unchanged table fragments extracted into templates and table files, and their two containing manuscript files. Each change is recorded explicitly. Tree hashes use sorted UTF-8 lines `SHA256  copied_to\n`, separately for original identities and maintained copies. `make verify` checks every maintained copy and both manifest trees without accessing upstream directories.

An additional archive comparison matched 121 source/configuration files byte-for-byte against the final LLM result ZIPs. Another 24 source-tree files (tests, project metadata, and two V2.3 configurations) were not members at those archive paths; their source-tree hashes are retained without claiming archive identity. Per-file results are in `manifests/llm_source_archive_identity.json`.

The immutable dataset revision is `b189ec765aa7ed75c8acfea42df31fdae71f97be`. Saved dataset/model manifests retain sample identities, prompts, model revision IDs, tokenizer/rendering hashes, weight inventories, and known access failures. Large rendered prompt files and model caches are not bundled. `external_design_inputs.tsv` records paths, sizes and freshly calculated hashes for the available external design files. `external_analysis_inputs.tsv` separately covers the 32 cached split/bootstrap/configuration files used by the tested analysis replay. Neither external manifest is needed by Level A.

## Vision, traffic and legacy BERT

The expected local ISQED checkout was absent. The public fallback `https://github.com/youlei202/ISQED` was fetched into external work storage at commit `5f94f5daf5f95d3d7d917c23b248308699dc30fc`. The two source blobs recorded by the paper were checked exactly: `exp11_shape_bias_texture.py` is `972b6ec4d35f4e9a661a8ffdabac5a940f727f69`, and `exp13_image_model_geometry.py` is `84cbf2af8a1bb3d61e25428b0789a222f0c9403f`. All selected source file hashes and the upstream MIT license are retained.

The frozen empirical inputs come from the uploaded paper's `tables(1).zip` and `artifacts(1).zip` identities, not from newly executed ISQED code. Their bytes match the package's identity manifest; the historical archives themselves were not separately recovered on this server. Vision residual arrays and context-specific PCA coordinates are preserved. Traffic values include infeasible coefficients, and no missing pruning curves were reconstructed. BERT evidence is explicitly legacy sentiment-classifier evidence.

## Adaptations and limits

Paper build scripts contain no server-specific absolute input path. Absolute paths remain in provenance and byte-preserved archival experiment snapshots. `stage_experiments.py` creates a fresh external workspace, changes only recorded path prefixes, pins the archived model/dataset revisions, and disables model-roster fallback; it records every changed hash in `PATH_MIGRATION.json`. It does not execute inference or reuse an old run directory. These adapters are syntax/design tested; full inference with them remains `not_run`.

No existing experiment source or result was modified. No public release, remote push, submission, model-weight copy, or inferred author identity was created. Licensing limits and unavailable historical inputs remain explicit in `THIRD_PARTY_NOTICES.md` and `docs/KNOWN_LIMITATIONS.md`.
