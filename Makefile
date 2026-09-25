SHELL := /bin/bash
.NOTPARALLEL:
.PHONY: help verify-inputs tables figures paper replay verify analysis clean environment

help:
	@printf '%s\n' 'verify-inputs  Check frozen hashes, schemas and completed vector audit' 'tables         Replay numeric tables and panel sources' 'figures        Build all standalone PDFs (also replays tables)' 'paper          Clean LaTeX/BibTeX manuscript build' 'replay         verify-inputs + tables + figures + paper; offline, no models' 'verify         Tests plus provenance, numeric, bibliography and PDF checks' 'analysis       Replay cached analyses; external raw root required' 'environment    Capture runtime versions' 'clean          Remove generated products; preserve frozen evidence'
verify-inputs:
	@bash scripts/verify_inputs.sh
tables:
	@bash scripts/build_tables.sh
figures: tables
	@bash scripts/build_figures.sh
paper:
	@bash scripts/build_paper.sh
replay: verify-inputs figures paper
verify:
	@bash scripts/verify.sh
analysis:
	@bash scripts/replay_analysis.sh
environment:
	@bash scripts/capture_environment.sh
clean:
	@bash scripts/clean.sh
