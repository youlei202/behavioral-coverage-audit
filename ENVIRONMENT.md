# Environments

## Level A: frozen-output replay

The tested packaging environment uses CPython 3.12.3. The exact resolved Python packages are recorded in `manifests/packaging_environment.lock.txt`; the smaller `requirements-replay.txt` contains the replay/test dependencies. Package versions were obtained from the actual installed environment, not inferred from the manuscript.

```bash
source scripts/env.sh
python3 -m venv "$BCA_WORK_ROOT/envs/replay"
"$BCA_WORK_ROOT/envs/replay/bin/python" -m pip install -r requirements-replay.txt
export BCA_PYTHON="$BCA_WORK_ROOT/envs/replay/bin/python"
```

The installation step may use the network. The replay targets do not. `scripts/env.sh` moves Python, pip and plotting caches and temporary files outside the source tree. `BCA_WORK_ROOT` defaults to a sibling work directory and can be set to any writable external directory. Build products are ignored by Git; frozen evidence is tracked.

Install pdfLaTeX and BibTeX before replay. The verified toolchain is TeX Live 2026, pdfTeX 1.40.29. Required packages include `standalone`, `pgfplots`, `newtx`, `caption`/`subcaption`, `multirow`, `makecell`, `enumitem`, `wrapfig`, `placeins`, `xstring`, `mweights`, `fontaxes`, and the standard/recommended LaTeX and font collections. One working TeX Live package installation is:

```bash
tlmgr install collection-latexrecommended collection-fontsrecommended \
  standalone pgfplots newtx caption multirow makecell enumitem wrapfig placeins \
  xstring mweights fontaxes fontspec realscripts anyfontsize
```

`scalefnt` and `centernot` are supplied by the installed `carlisle` and `oberdiek` packages. Local conference, bibliography, `natbib`, and header styles are preserved under `paper/styles/`; the build supplies their search paths. No fonts, margins, or scientific text were changed to force pagination. A Tectonic/XeTeX probe moved references to page 11; it is not the verified manuscript engine.

Run `make environment` to capture Python, pip, pdfLaTeX, BibTeX, and read-only `nvidia-smi` output under `build/environment/`. The committed packaging capture is under `manifests/environment/`. GPU output is an observation of the packaging host, not evidence of executed inference.

## Level B: cached analysis

Install `requirements-analysis.txt` in the replay environment or use a separate environment with the captured versions. The tested reporting replay uses CVXPY 1.7.4 for imports shared with the archived analysis modules; it does not invoke optimizers. The replay does not import or load Torch/Transformers models.

The original V2.2 lock remains in `experiments/llm/manifests/v22/env/requirements.lock.txt`. It used Python 3.11 and, among other packages, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, PyArrow 22.0.0 and CVXPY 1.7.4. The packaging environment differs; cached numerical tables were independently compared against the original frozen tables. This is not a bitwise-identical historical runtime claim.

## Full modern-LLM inference

Use the historical Python 3.11 environment and the copied V2 lock at `experiments/llm/manifests/v2/env/requirements.lock.txt`, not the Level-A requirements. The recorded inference stack uses Torch 2.9.1+cu128, Transformers 4.57.3, CUDA 12.8, and eight NVIDIA B200 devices. The saved environment manifests and runtime-repair notes remain under `experiments/llm/manifests/`.

V2.3 was prestaged with Python 3.11.13, while its final CPU recovery used 3.11.15. The saved recovery explicitly notes that the runtime was not bitwise identical, despite the same locked packages and scientific inputs. The final authorized numerical patch and completed markers are retained. Full model inference and full solver/bootstrap re-execution were not run during repository packaging.

## Legacy ISQED

The recovered source provides an unpinned `experiments/shared/ISQED_requirement.txt`. It includes older NLP dependencies, and does not freeze a complete Torch/TorchVision runtime. Keep this environment separate: its `protobuf<=3.20.0` constraint differs from the modern-LLM stack. Exact historical package versions, model revisions, and some data/checkpoint inventories were not available and have not been guessed. Source-derived entry points are documented with those limits.
