from common import ROOT
from verify_outputs import figure_sources

def test_one_panel_per_file_and_complete_provenance():
    rows,references=figure_sources()
    assert sum(r['kind']=='conceptual' for r in rows)==2
    assert len(set(references))==26

def test_active_build_scripts_are_portable():
    for path in (ROOT/'scripts').iterdir():
        if path.suffix in ['.py','.sh']:
            text=path.read_text()
            assert '/work/Users/leiyo/' not in text, path
            assert '/home/leiyo/.codex/attachments/' not in text, path
