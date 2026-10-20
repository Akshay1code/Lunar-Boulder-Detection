"""Download pinned official Mars train/val files only; never train or overwrite."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'datasets/mmlsv2'
REV = 'b2cb0bec0306c9ef7ea7690041c75a5e029c61bf'
API = 'https://huggingface.co/api/datasets/MarsLS/MMLSv2/tree/' + REV
RESOLVE = 'https://huggingface.co/datasets/MarsLS/MMLSv2/resolve/' + REV + '/'

def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def verify(data, entry):
    if len(data) != entry['size']:
        raise ValueError('Size mismatch: ' + entry['path'])
    if 'lfs' in entry:
        actual = hashlib.sha256(data).hexdigest()
        expected = entry['lfs']['oid']
    else:
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        expected = entry['oid']
    if actual != expected:
        raise ValueError('Published hash mismatch: ' + entry['path'])

def download(entry):
    path = DEST / entry['path']
    if path.exists():
        data = path.read_bytes()
        verify(data, entry)  # Never replace an existing mismatched file.
    else:
        data = fetch(RESOLVE + entry['path'])
        verify(data, entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    return dict(path=entry['path'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def main():
    entries = []
    for split, count in [('train', 465), ('val', 66)]:
        listing = json.loads(fetch(API + '/' + split + '?recursive=true&limit=1000'))
        files = [e for e in listing if e['type'] == 'file']
        images = {Path(e['path']).name for e in files if e['path'].startswith(split+'/images/')}
        masks = {Path(e['path']).name for e in files if e['path'].startswith(split+'/masks/')}
        if images != masks or len(images) != count or len(files) != count*2:
            raise ValueError('Unexpected pair listing: ' + split)
        entries.extend(files)
    size = sum(e['size'] for e in entries)
    if shutil.disk_usage(ROOT).free < size*2 + 1024**3:
        raise RuntimeError('Insufficient free disk space')
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = DEST/'source_manifest.json'
    if not manifest.exists():
        with manifest.open('x', encoding='utf-8') as out:
            json.dump(dict(repository='MarsLS/MMLSv2', revision=REV, bytes=size, files=entries),out,indent=2)
    else:
        if json.loads(manifest.read_text())['files'] != entries:
            raise RuntimeError('Existing manifest differs')
    for name,url in [('README.source.md',RESOLVE+'README.md'),
                     ('LICENSE.github.txt','https://raw.githubusercontent.com/MAIN-Lab/MMLS_v2/main/LICENSE')]:
        path=DEST/name
        if not path.exists():
            with path.open('xb') as out:
                out.write(fetch(url))
    print(f'Downloading/verifying {len(entries)} files, {size} bytes', flush=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,e) for e in entries]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results)%100==0:
                print(f'Verified {len(results)}/{len(entries)}',flush=True)
    inventory=DEST/'download_inventory.json'
    if not inventory.exists():
        with inventory.open('x',encoding='utf-8') as out:
            json.dump(sorted(results,key=lambda r:r['path']),out,indent=2)
    print('Download complete; all sizes and published Git/LFS hashes verified.',flush=True)

if __name__ == '__main__':
    main()
