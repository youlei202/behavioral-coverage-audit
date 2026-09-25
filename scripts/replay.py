"""Offline Level-A orchestration around the unmodified uploaded replay scripts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
from common import ROOT, data_map, require, sha256, read_tsv

BUILD=ROOT/'build'
STAGE=BUILD/'replay'

def run(command,cwd,log,env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as out:
        result=subprocess.run(command,cwd=cwd,stdout=out,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        print(log.read_text(errors='replace')[-7000:],file=sys.stderr)
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')

def tables():
    from verify_inputs import verify
    from generate_inline_tables import generate
    verify()
    if STAGE.exists():shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for folder in ['sections','appendix','tables','figures/standalone']:
        shutil.copytree(ROOT/'paper'/folder,STAGE/folder)
    for name in ['main.tex','references.bib','figures/figure_style.tex']:
        shutil.copyfile(ROOT/'paper'/name,STAGE/name)
    for f in (ROOT/'paper/styles').iterdir():shutil.copyfile(f,STAGE/f.name)
    (STAGE/'scripts').mkdir()
    for name in ['build_assets.py','build_vector_evidence.py','read_scalar_parquet.py']:
        shutil.copyfile(ROOT/'scripts'/name,STAGE/'scripts'/name)
    for old,new in data_map().items():
        target=STAGE/old;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/new,target)
    (STAGE/'qa').mkdir()
    for name in ['FIGURE_PROVENANCE.md','SOURCE_MAP.md']:
        shutil.copyfile(ROOT/'manifests/paper_package'/name,STAGE/name)
    for name in ['build_assets.py','build_vector_evidence.py']:
        run([sys.executable,str(STAGE/'scripts'/name)],STAGE,BUILD/'logs'/f'{name}.log')
    generate(STAGE)
    generated=ROOT/'data/generated';generated.mkdir(exist_ok=True)
    shutil.copytree(STAGE/'data/figures',generated/'figures',dirs_exist_ok=True)
    shutil.copyfile(STAGE/'data/v22/vector_audit/vector_summary.csv',generated/'vector_summary.csv')
    for folder in ['tables','figures/standalone']:
        shutil.copytree(STAGE/folder,ROOT/'paper'/folder,dirs_exist_ok=True)
    shutil.copyfile(STAGE/'figures/figure_style.tex',ROOT/'paper/figures/figure_style.tex')
    shutil.copytree(STAGE/'qa',BUILD/'qa',dirs_exist_ok=True)
    numeric()
    print('Tables and all 26 panel sources regenerated from frozen inputs')

def numeric():
    expected=read_tsv(ROOT/'manifests/paper_expected_outputs.tsv')
    for row in expected:
        path=ROOT/row['relative_path']
        require(path.is_file(),f'Missing replay output: {path}')
        require(sha256(path)==row['sha256'],f'Replay baseline differs: {path}')
    print(f'Generated text/numeric baseline verified: {len(expected)} outputs')

def figures():
    require((STAGE/'data/figures').is_dir(),'Run make tables before make figures')
    require(shutil.which('pdflatex') is not None,'pdfLaTeX is required; see ENVIRONMENT.md')
    output=BUILD/'latex/panels';output.mkdir(parents=True,exist_ok=True)
    sources=sorted((ROOT/'paper/figures/standalone').glob('*.tex'))
    def compile_one(path):
        name=path.stem
        run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',
             '-file-line-error','-output-directory='+str(output),str(path)],
             STAGE/'figures/standalone',BUILD/'logs'/f'{name}.compile.txt')
        target=ROOT/'paper/figures/pdf'/f'{name}.pdf'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(output/f'{name}.pdf',target)
        return name
    with ThreadPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
        list(pool.map(compile_one,sources))
    print(f'Compiled {len(sources)} standalone panels using pdfLaTeX')

def paper():
    from check_bibliography import check
    check()
    for name in ['pdflatex','bibtex']:
        require(shutil.which(name) is not None,f'{name} is required; see ENVIRONMENT.md')
    for source in (ROOT/'paper/figures/standalone').glob('*.tex'):
        require((ROOT/'paper/figures/pdf'/f'{source.stem}.pdf').is_file(),'Run make figures before make paper')
    output=BUILD/'latex/manuscript'
    if output.exists():shutil.rmtree(output)
    output.mkdir(parents=True)
    env=dict(os.environ)
    env['TEXINPUTS']=str(ROOT/'paper/styles')+os.pathsep+env.get('TEXINPUTS','')
    env['BIBINPUTS']=str(ROOT/'paper')+os.pathsep+env.get('BIBINPUTS','')
    env['BSTINPUTS']=str(ROOT/'paper/styles')+os.pathsep+env.get('BSTINPUTS','')
    cmd=['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',
         '-file-line-error','-output-directory='+str(output),'main.tex']
    run(cmd,ROOT/'paper',BUILD/'logs/main-pass1.txt',env)
    run(['bibtex','main'],output,BUILD/'logs/bibtex.txt',env)
    for number in [2,3]:run(cmd,ROOT/'paper',BUILD/'logs'/f'main-pass{number}.txt',env)
    shutil.copyfile(output/'main.pdf',ROOT/'paper/main.pdf')
    print('Built paper/main.pdf from clean LaTeX/BibTeX intermediates')

def clean():
    for path in [BUILD,ROOT/'data/generated']:
        if path.exists():shutil.rmtree(path)
    (ROOT/'data/generated').mkdir()
    (ROOT/'paper/main.pdf').unlink(missing_ok=True)
    for f in (ROOT/'paper/figures/pdf').glob('*.pdf'):f.unlink()
    print('Removed generated data and build products; frozen evidence and LaTeX sources retained')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['tables','figures','paper','numeric','clean'])
    arguments=parser.parse_args()
    globals()[arguments.operation]()
