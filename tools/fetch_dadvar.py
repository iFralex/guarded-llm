#!/usr/bin/env python3
"""Download the authors' public CC BY 4.0 research dataset; retain it locally."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

url = 'https://zenodo.org/api/records/10143097/files/Cyberbullying_YouTube_Dadvar_2015.zip/content'
expected = 'a0271497653cc8907052e57d8ff9bc0cc22faab461bc3c92355b1fc14cca9450'
path = Path(__file__).resolve().parents[1] / 'data/external/dadvar-youtube.zip'
path.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url, timeout=120) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit('Dataset checksum mismatch: refusing to save')
path.write_bytes(data)
print('Downloaded and verified', path)
