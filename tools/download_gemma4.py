#!/usr/bin/env python3
"""Resume the pinned Gemma artifact using four verified HTTP ranges."""
import concurrent.futures, hashlib, os, sys, subprocess
from pathlib import Path

dest=Path(sys.argv[1]); size=2588147712; sha='181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c'
url='https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/6b78abd019e61a1ca4cbe3b212d2c9ce8ff38a94/gemma-4-E2B-it.litertlm'
have=dest.stat().st_size if dest.exists() else 0
if have>=size: raise SystemExit('Destination already full or larger; verify it rather than overwriting')
remaining=size-have; span=(remaining+3)//4
ranges=[(have+i*span,min(size-1,have+(i+1)*span-1)) for i in range(4) if have+i*span<size]
parts=[dest.with_name(dest.name+f'.part{i}') for i in range(len(ranges))]
if any(p.exists() for p in parts): raise SystemExit('Partial range files already exist; inspect before restarting')
def fetch(item):
    (start,end),path=item
    headers=path.with_name(path.name+'.headers')
    subprocess.run(['curl','-fsSL','--retry','2','-r',f'{start}-{end}','-D',str(headers),'-o',str(path),url],check=True)
    header_text=headers.read_text(errors='replace').lower()
    if f'content-range: bytes {start}-{end}/' not in header_text:
        raise RuntimeError('Unexpected HTTP range response')
    if path.stat().st_size!=end-start+1: raise RuntimeError('Incomplete range')
    print('Completed range',start,end,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch,zip(ranges,parts)))
with dest.open('ab') as out:
    for part in parts:
        with part.open('rb') as inp:
            while True:
                block=inp.read(1024*1024)
                if not block: break
                out.write(block)
if dest.stat().st_size!=size: raise RuntimeError('Combined size mismatch')
h=hashlib.sha256()
with dest.open('rb') as inp:
    for block in iter(lambda:inp.read(8*1024*1024),b''): h.update(block)
if h.hexdigest()!=sha: raise RuntimeError('Combined SHA-256 mismatch; preserving files for inspection')
for part in parts:
    part.unlink()
    part.with_name(part.name+'.headers').unlink()
print('Verified SHA-256',h.hexdigest(),flush=True)
