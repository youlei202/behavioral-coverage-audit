from __future__ import annotations

import zipfile

from pier_llm.packaging import PACKAGE_NAME, build_results_package, validate_results_package
from pier_llm.plotting import make_all_figures
from pier_llm.smoke import _tiny_plot_frames
from pier_llm.utils import atomic_write_text


def test_all_required_figures_and_package(tmp_path) -> None:
    frames = _tiny_plot_frames()
    paths = make_all_figures(tmp_path / "outputs" / "figures", **frames)
    assert len(paths) == 9
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    atomic_write_text(tmp_path / "outputs/analysis/GOLDMINE_REPORT.md", "# report\n")
    atomic_write_text(tmp_path / "REPRODUCE.md", "# reproduce\n")
    atomic_write_text(tmp_path / "outputs/raw_scores/test.txt", "raw\n")
    atomic_write_text(tmp_path / "src/pier_llm/__pycache__/ignored.pyc", "cache\n")
    package = build_results_package(tmp_path)
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert f"{PACKAGE_NAME}/GOLDMINE_REPORT.md" in names
    assert f"{PACKAGE_NAME}/REPRODUCE.md" in names
    assert f"{PACKAGE_NAME}/MANIFEST.sha256" in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    validation = validate_results_package(package)
    assert validation["manifest_entry_count"] == len(names) - 1
