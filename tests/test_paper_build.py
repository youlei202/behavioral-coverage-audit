from check_bibliography import check
from verify_outputs import verify

def test_only_cited_cleaned_bibliography_entries():
    assert len(check())==23

def test_clean_manuscript_and_panel_builds():
    result=verify()
    assert result['references_first_page']==10
    assert result['tables']==10
    assert result['composite_figures']==8
