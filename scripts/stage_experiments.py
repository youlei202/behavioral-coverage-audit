"""Stage a fresh, portable experiment workspace; does not execute experiments."""
from pathlib import Path
import argparse
import json
import shutil
from common import ROOT, read_tsv, require, sha256

STAGES={'v2':('inference','PIER_LLM_ECOSYSTEM_GOLDMINE_V2'),
        'v21':('analysis','PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS'),
        'v22':('analysis','PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS'),
        'v23':('analysis','PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION')}

def stage(destination,raw_root=None,model_cache=None):
    destination=destination.resolve()
    require(not destination.exists(),'Run root already exists; use a new directory, never overwrite a frozen run')
    require(not destination.is_relative_to(ROOT),'Place experiment work outside the source repository')
    records=read_tsv(ROOT/'manifests/upstream_sources.tsv')
    replacements={}
    for short,(group,name) in STAGES.items():
        # Derive former root names from recorded provenance, not server constants in code.
        match=next(r for r in records if r['component']=='llm-'+short and '/src/' in r['upstream_path_or_repo'])
        physical=match['upstream_path_or_repo'].split('/src/')[0]
        config=ROOT/'experiments/llm'/group/short/'configs'
        if short=='v2':logical=json.loads((config/'experiment.json').read_text())['project_root']
        elif short=='v21':
            options=list(config.glob('*.json'))
            logical=json.loads(options[0].read_text()).get('logical_root',physical)
        elif short=='v22':logical=json.loads((config/'reanalysis_v22.json').read_text())['logical_root']
        else:logical=json.loads((config/'experiment_v23.json').read_text())['logical_root']
        replacements[physical]=str(destination/name)
        replacements[logical]=str(destination/name)
    destination.mkdir(parents=True)
    changes=[]
    for short,(group,name) in STAGES.items():
        upstream=ROOT/'experiments/llm'/group/short
        target=destination/name
        shutil.copytree(upstream,target)
        for folder in ['env','logs','artifacts','status']:(target/folder).mkdir(exist_ok=True)
        stored_env=ROOT/'experiments/llm/manifests'/short/'env'
        if stored_env.exists():shutil.copytree(stored_env,target/'env',dirs_exist_ok=True)
        if model_cache is not None:
            (target/'hf_cache').symlink_to(model_cache.resolve(),target_is_directory=True)
        for file in sorted(target.rglob('*')):
            if file.is_symlink() or not file.is_file() or file.suffix not in ['.py','.sh','.json','.toml','.txt','.md','.yaml']:continue
            before=sha256(file);text=file.read_text()
            # Longest prefix first: V2 is a prefix of V2.1/V2.2/V2.3 directory names.
            for old,new in sorted(replacements.items(),key=lambda item:-len(item[0])):text=text.replace(old,new)
            file.write_text(text)
            if sha256(file)!=before:changes.append(dict(path=str(file.relative_to(destination)),before_sha256=before,after_sha256=sha256(file),change='path prefixes only'))
    legacy=destination/'ISQED';(legacy/'experiments').mkdir(parents=True)
    shutil.copytree(ROOT/'experiments/shared/isqed',legacy/'isqed')
    for family in ['vision','traffic','bert_legacy']:
        for file in (ROOT/'experiments'/family/'src').glob('*.py'):
            shutil.copyfile(file,legacy/'experiments'/file.name)
    (legacy/'experiments/__init__.py').touch()
    if raw_root is not None:
        # Only verified non-model inputs used to prepare future inference are copied.
        for row in read_tsv(ROOT/'manifests/external_design_inputs.tsv'):
            source=raw_root/row['relative_path']
            require(source.is_file() and sha256(source)==row['sha256'],f'Frozen design SHA256 mismatch/missing: {source}')
            target=destination/row['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            if target.suffix=='.json':
                before=sha256(target);text=target.read_text()
                for old,new in sorted(replacements.items(),key=lambda item:-len(item[0])):text=text.replace(old,new)
                target.write_text(text)
                if sha256(target)!=before:changes.append(dict(path=str(target.relative_to(destination)),before_sha256=before,after_sha256=sha256(target),change='metadata path prefixes only'))
    # Pin the recorded revisions when fresh prestaging must retrieve missing assets.
    # The archived entry point otherwise resolves the current remote HEAD.
    v2=destination/STAGES['v2'][1]
    resolved=json.loads((v2/'configs/resolved_models.json').read_text())['models']
    revisions={m['id']:m['revision'] for m in resolved}
    catalog_path=v2/'configs/models.json'
    before=sha256(catalog_path);catalog=json.loads(catalog_path.read_text())
    require({m['id'] for m in catalog['preferred']}==set(revisions),'Frozen roster differs from preferred models')
    for model in catalog['preferred']:model['revision']=revisions[model['id']]
    catalog['fallback']=[]
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
    changes.append(dict(path=str(catalog_path.relative_to(destination)),before_sha256=before,after_sha256=sha256(catalog_path),change='pin recorded model revisions; forbid roster fallback'))
    dataset=json.loads((ROOT/'experiments/llm/manifests/v2/data/manifests/dataset_source.json').read_text())
    config_path=v2/'configs/experiment.json'
    before=sha256(config_path);config=json.loads(config_path.read_text())
    config['dataset_revision']=dataset['dataset_revision']
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    changes.append(dict(path=str(config_path.relative_to(destination)),before_sha256=before,after_sha256=sha256(config_path),change='pin recorded dataset revision'))
    for file,old,new in [
        (v2/'src/pier_llm/model_prestage.py','api.model_info(spec["id"], files_metadata=True)',
         'api.model_info(spec["id"], revision=spec["revision"], files_metadata=True)'),
        (v2/'src/pier_llm/data_prep.py','api.dataset_info(config["dataset_id"])',
         'api.dataset_info(config["dataset_id"], revision=config["dataset_revision"])')]:
        before=sha256(file);text=file.read_text()
        require(text.count(old)==1,f'Cannot apply recorded-revision adapter: {file}')
        file.write_text(text.replace(old,new))
        changes.append(dict(path=str(file.relative_to(destination)),before_sha256=before,after_sha256=sha256(file),change='use recorded immutable revision in retrieval; no estimator change'))
    (destination/'PATH_MIGRATION.json').write_text(json.dumps(dict(replacements=replacements,changed_files=changes,
        model_cache_copied=False,experiments_executed=False,raw_design_copied=raw_root is not None),indent=2)+'\n')
    print(f'Staged source/configuration in {destination}; no experiment was executed')
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--model-cache',type=Path)
    args=parser.parse_args()
    stage(args.run_root,args.raw_root,args.model_cache)
