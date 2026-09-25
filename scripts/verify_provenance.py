"""Check maintained copies against their recorded upstream identities, offline."""
import hashlib
import json
from common import ROOT, owned_path, read_tsv, require, sha256


def tree_hash(rows, field):
    content=''.join(row[field]+'  '+row['copied_to']+'\n'
                    for row in sorted(rows,key=lambda row:row['copied_to']))
    return hashlib.sha256(content.encode()).hexdigest()


def verify(root=ROOT):
    rows=read_tsv(root/'manifests/upstream_sources.tsv')
    require(len({r['copied_to'] for r in rows})==len(rows),'Duplicate source destinations')
    for row in rows:
        path=owned_path(row['copied_to'],root)
        require(path.is_file(),f'Missing maintained upstream copy: {path}')
        require(path.stat().st_size==int(row['maintained_size_bytes']) and
                sha256(path)==row['maintained_sha256'],f'Maintained source identity changed: {path}')
        if row['sha256']!=row['maintained_sha256']:
            require(row['notes'].startswith('adapted:'),f'Unrecorded source adaptation: {path}')
    trees=json.loads((root/'manifests/source_tree_hashes.json').read_text())
    require(set(trees)=={r['component'] for r in rows},'Source tree component mismatch')
    for component,record in trees.items():
        selected=[r for r in rows if r['component']==component]
        require(record['files']==len(selected),'Source tree file count mismatch')
        require(record['upstream_identity_tree_sha256']==tree_hash(selected,'sha256'),
                f'Upstream identity tree changed: {component}')
        require(record['maintained_tree_sha256']==tree_hash(selected,'maintained_sha256'),
                f'Maintained source tree changed: {component}')
    return len(rows)


if __name__=='__main__':
    print(f'Maintained source provenance verified: {verify()} files')
