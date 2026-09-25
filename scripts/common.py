"""Repository-relative paths and explicit integrity checks."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read_tsv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def data_map(root=ROOT):
    return json.loads((root/'manifests/data_map.json').read_text())

def frozen(original_path, root=ROOT):
    return root/data_map(root)[original_path]

def owned_path(relative, root=ROOT):
    path=(root/relative).resolve()
    require(path.is_relative_to(root.resolve()), f'Path escapes repository: {relative}')
    return path
