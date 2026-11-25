#!/usr/bin/env python3
"""Download and decrypt the authors' public eSPD MEGA archive, verifying SHA-256."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

FOLDER = 'SZMnDQaY'
FOLDER_KEY = 'E5NXBL8uaYAhc5ebZEeuEg'
API = f'https://g.api.mega.co.nz/cs?id=12586&n={FOLDER}'


def decode(value):
    return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))


def request(command):
    response = requests.post(API, json=[command], timeout=60)
    response.raise_for_status()
    result = response.json()[0]
    if not isinstance(result, dict):
        raise RuntimeError(f'MEGA API error: {result}')
    return result


def entries():
    for node in request({'a': 'f', 'c': 1, 'r': 1})['f']:
        if node['t'] != 0:
            continue
        encrypted_key = decode(node['k'].split(':')[-1].split('/')[0])
        cipher = Cipher(algorithms.AES(decode(FOLDER_KEY)), modes.ECB()).decryptor()
        raw_key = cipher.update(encrypted_key) + cipher.finalize()
        key = bytes(a ^ b for a, b in zip(raw_key[:16], raw_key[16:]))
        cipher = Cipher(algorithms.AES(key), modes.CBC(bytes(16))).decryptor()
        attrs = (cipher.update(decode(node['a'])) + cipher.finalize()).rstrip(b'\0')
        if not attrs.startswith(b'MEGA'):
            raise RuntimeError('Failed to decode MEGA metadata')
        yield json.loads(attrs[4:])['n'], node, key, raw_key[16:24] + bytes(8)


def download(node, key, iv, path):
    response = requests.get(request({'a': 'g', 'g': 1, 'n': node['h']})['g'], stream=True, timeout=(30, 120))
    response.raise_for_status()
    decrypt = Cipher(algorithms.AES(key), modes.CTR(iv)).decryptor()
    digest = hashlib.sha256()
    count = 0
    with path.open('xb') as output:
        for block in response.iter_content(chunk_size=1024 * 1024):
            data = decrypt.update(block)
            output.write(data)
            digest.update(data)
            count += len(data)
            if count // (256 * 1024 * 1024) != (count - len(data)) // (256 * 1024 * 1024):
                print(f'Decrypted {count // (1024 * 1024)} MiB', flush=True)
        tail = decrypt.finalize()
        output.write(tail)
        digest.update(tail)
    if count != node['s']:
        raise RuntimeError(f'Incomplete download: {count} of {node["s"]} bytes')
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    files = {name: (node, key, iv) for name, node, key, iv in entries()}
    checksum = args.directory / 'results.tar.gz.sha256'
    archive = args.directory / 'results.tar.gz'
    if checksum.exists() or archive.exists():
        parser.error('Output already exists')
    download(*files['results.tar.gz.sha256'], checksum)
    expected = checksum.read_text().split()[0]
    actual = download(*files['results.tar.gz'], archive)
    if actual != expected:
        raise RuntimeError(f'SHA-256 mismatch: {actual} != {expected}')
    print(f'Archive verified: {actual}', flush=True)


if __name__ == '__main__':
    main()
