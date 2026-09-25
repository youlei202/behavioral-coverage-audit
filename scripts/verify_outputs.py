"""Verify all paper outputs and record only checks performed in this checkout."""
from pathlib import Path
import json
import re
import pymupdf
from common import ROOT, read_tsv, require, sha256
from replay import numeric
from check_bibliography import check
from verify_provenance import verify as verify_provenance

def figure_sources(root=ROOT):
    rows=read_tsv(root/'manifests/figure_provenance.tsv')
    sources=sorted((root/'paper/figures/standalone').glob('*.tex'))
    require(len(rows)==len(sources)==26,'Standalone panel denominator changed')
    require({r['panel'] for r in rows}=={p.stem for p in sources},'Panel provenance is incomplete')
    for row in rows:
        text=(root/row['source']).read_text()
        require(len(re.findall(r'\\begin\{tikzpicture\}',text))==1,f'Expected one TikZ picture: {row["panel"]}')
        inputs=[p for p in row['frozen_inputs'].split(';') if p]
        require(bool(inputs)==(row['kind']=='empirical'),f'Conceptual/empirical mismatch: {row["panel"]}')
        for path in inputs:require((root/path).is_file(),f'Missing figure source input: {path}')
        if row['kind']=='conceptual':
            require('data/figures/' not in text,'Conceptual panel refers to measured coordinates')
    tex='\n'.join(f.read_text() for f in (root/'paper').rglob('*.tex'))
    references=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex)
    require(set(Path(x).stem for x in references)=={p.stem for p in sources},'Included figures differ from panel manifest')
    return rows,references

def verify(root=ROOT):
    # Baseline is generated evidence from the upload, never an original measurement.
    numeric()
    source_files=verify_provenance(root)
    cited=check(root)
    rows,references=figure_sources(root)
    for relative in references:
        path=root/'paper'/relative
        require(path.is_file(),f'Missing figure PDF: {relative}')
        with pymupdf.open(path) as doc:require(len(doc)==1,f'Panel has multiple pages: {relative}')
    log=root/'build/latex/manuscript/main.log'
    require(log.is_file(),'No current clean manuscript build log')
    final_log=log.read_text(errors='replace')
    bad=re.findall(r'(?:LaTeX|Package natbib) Warning:.*(?:undefined|multiply defined)|There were undefined|^! .+',final_log,re.M)
    require(not bad,f'LaTeX diagnostics: {bad}')
    require(not re.search(r'LaTeX Warning: File .* not found',final_log),'LaTeX reports missing files')
    biblog=(root/'build/latex/manuscript/main.blg').read_text(errors='replace')
    require('Warning--' not in biblog and 'error message' not in biblog.lower(),'BibTeX reports warnings/errors')
    bbl=(root/'build/latex/manuscript/main.bbl').read_text()
    items=set(re.findall(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',bbl))
    require(items==cited,'Rendered bibliography differs from cited keys')
    with pymupdf.open(root/'paper/main.pdf') as doc:
        text=[p.get_text() for p in doc]
        refs=[i+1 for i,t in enumerate(text) if re.search(r'(?m)^REFERENCES\s*$|^References\s*$',t)]
        require(refs==[10],f'References must begin on page 10, observed {refs}')
        require('DISCUSSION' in text[8].upper(),'Scientific main text no longer fills page 9')
        pages=len(doc)
    qa=json.loads((root/'build/qa/numeric_checks.json').read_text())
    reference=json.loads((root/'manifests/paper_package/qa/numeric_checks.json').read_text())
    require(qa==reference,'Arithmetic QA differs from uploaded baseline')
    aux=(root/'build/latex/manuscript/main.aux').read_text()
    table_labels=set(re.findall(r'\\newlabel\{(tab:[^}]+)\}',aux))
    figure_labels=set(re.findall(r'\\newlabel\{(fig:[^}]+)\}',aux))
    require(len(table_labels)==10,f'Table count changed: {len(table_labels)}')
    require(len(figure_labels)==8,f'Composite figure count changed: {len(figure_labels)}')
    paper_items=json.loads((root/'manifests/paper_items.json').read_text())
    require(len(paper_items)==18 and {item['label'] for item in paper_items}==table_labels|figure_labels,
            'Figure/table provenance does not account for every manuscript item')
    frozen_paths={row['relative_path'] for row in read_tsv(root/'manifests/input_sha256.tsv')}
    for item in paper_items:
        match=re.search(r'\\newlabel\{'+re.escape(item['label'])+r'\}\{\{([^}]+)\}\{([^}]+)\}',aux)
        require(match and item['item'].split()[-1]==match[1] and item['page']==int(match[2]),
                f'Manuscript item numbering/page changed: {item["label"]}')
        require((root/item['source']).is_file() and (root/item['generator']).is_file(),
                f'Missing item source/generator: {item["label"]}')
        require(bool(item['frozen_inputs'])==(item['family']!='conceptual'),
                f'Empirical/conceptual item provenance mismatch: {item["label"]}')
        require(set(item['frozen_inputs'])<=frozen_paths,f'Unmanifested item inputs: {item["label"]}')
    result=dict(level_a='passed',unique_panels=len(rows),empirical_panels=sum(r['kind']=='empirical' for r in rows),
                composite_figures=len(figure_labels),tables=len(table_labels),pages=pages,
                references_first_page=refs[0],citations=len(cited),undefined_citations=0,undefined_references=0,
                new_inference=False,new_fits=False,new_bootstrap=False,pdf_sha256=sha256(root/'paper/main.pdf'),
                numeric_qa=qa,maintained_source_files=source_files,provenance_items=len(paper_items))
    (root/'build/qa/replay_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Outputs verified: {len(rows)} panels, {len(figure_labels)} figures, {len(table_labels)} tables; references on page 10')
    return result

if __name__=='__main__':verify()
