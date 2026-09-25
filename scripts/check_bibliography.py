"""Check all manuscript citation keys against the supplied cleaned bibliography."""
import re
from common import ROOT, require, sha256
import json

def check(root=ROOT):
    text='\n'.join(re.sub(r'(?<!\\)%[^\n]*','',p.read_text()) for p in (root/'paper').rglob('*.tex'))
    cited={key.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',text) for key in group.split(',')}
    bib=root/'paper/references.bib'
    entries=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib.read_text())
    require(len(entries)==len(set(entries)), 'Duplicate bibliography keys')
    require(cited==set(entries),f'Bibliography mismatch: missing={sorted(cited-set(entries))}, uncited={sorted(set(entries)-cited)}')
    identity=json.loads((root/'manifests/bootstrap_identity.json').read_text())
    require(sha256(bib)==identity['cleaned_bibliography_sha256'],'Cleaned bibliography bytes changed')
    print(f'Bibliography verified: {len(cited)} cited entries, no missing or unused keys')
    return cited

if __name__=='__main__':check()
