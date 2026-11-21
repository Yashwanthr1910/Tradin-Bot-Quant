"""Small provenance helpers for reproducible saved research runs."""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess

def sha256_file(path:str|Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def git_state(root:Path|str=".")->dict:
    try:
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
        dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,text=True,check=True).stdout.strip())
        return {"commit":commit,"working_tree_dirty":dirty}
    except (OSError,subprocess.CalledProcessError):
        return {"commit":None,"working_tree_dirty":None}
