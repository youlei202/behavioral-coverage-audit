"""Create a fresh source-only copy and run Level A without upstream input paths."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
from common import ROOT, require, sha256

def run(destination):
    destination=destination.resolve()
    require(not destination.exists(),'Clean-room destination already exists')
    require(not destination.is_relative_to(ROOT),'Clean-room copy must be outside the source repository')
    ignore=shutil.ignore_patterns('.git','build','generated','__pycache__','.pytest_cache','*.pyc','*.pdf',
                                  '*.aux','*.log','*.out','*.blg','*.bbl','*.fls','*.fdb_latexmk','*.compile.txt')
    shutil.copytree(ROOT,destination,ignore=ignore)
    (destination/'data/generated').mkdir(exist_ok=True)
    require(not list(destination.rglob('*.pdf')),'A compiled PDF leaked into the clean room')
    require(not (destination/'build').exists(),'Build products leaked into the clean room')
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.pop('BEHAVIORAL_COVERAGE_RAW_DATA',None)
    env.pop('BEHAVIORAL_COVERAGE_MODEL_CACHE',None)
    before={str(f.relative_to(destination)):sha256(f) for f in (destination/'data/frozen').rglob('*') if f.is_file()}
    namespace=shutil.which('unshare')
    isolated=bool(namespace and subprocess.run([namespace,'-Urn','true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0)
    commands=[]
    for target in ['replay','verify']:
        command=['make',target]
        if isolated:command=[namespace,'-Urn','--',*command]
        subprocess.run(command,cwd=destination,env=env,check=True)
        commands.append('make '+target)
    after={str(f.relative_to(destination)):sha256(f) for f in (destination/'data/frozen').rglob('*') if f.is_file()}
    require(before==after,'Clean-room replay changed frozen evidence')
    replay_result=json.loads((destination/'build/qa/replay_verification.json').read_text())
    subprocess.run(['make','clean'],cwd=destination,env=env,check=True)
    commands.append('make clean')
    after_clean={str(f.relative_to(destination)):sha256(f) for f in (destination/'data/frozen').rglob('*') if f.is_file()}
    require(before==after_clean,'Cleaning changed frozen evidence')
    require(not (destination/'build').exists() and not list(destination.rglob('*.pdf')),
            'Cleaning left build products behind')
    result=dict(status='passed',commands=commands,source_only_initial_copy=True,
                no_initial_pdfs=True,external_raw_variables_removed=True,pythonpath_removed=True,
                frozen_evidence_unchanged=True,clean_preserves_frozen_evidence=True,
                clean_removes_generated_products=True,network_namespace_isolated=isolated,
                network_note='Replay contains no downloads/network calls; namespace result is reported separately',
                result=replay_result)
    (ROOT/'build/qa/clean_room_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Clean-room replay and verification passed:',destination)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    run(args.destination)
