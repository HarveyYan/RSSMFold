import os
import pathlib
import requests
import RSSMFold
from tqdm import tqdm
import hashlib
import urllib

single_seq_rssm_weights_link = 'https://www.dropbox.com/s/b3kavezmyq94t8j/single_seq_rssm_weights.pkl?dl=0'
cov_rssm_weights_link = 'https://www.dropbox.com/s/26uh7b81g2hc3gm/cov_rssm_weights.pkl?dl=0'
msa_rssm_weights_link = 'https://www.dropbox.com/s/iapa4a2mp2ykjk8/msa_rssm_weightsl.pkl?dl=0'

basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()
rssm_weights_dir = os.path.join(basedir, 'RSSMFold', 'rssm_weights')
single_seq_rssm_weights_path = os.path.join(rssm_weights_dir, 'single_seq_rssm_weights.pkl')
cov_rssm_weights_path = os.path.join(rssm_weights_dir, 'cov_rssm_weights.pkl')
msa_rssm_weights_path = os.path.join(rssm_weights_dir, 'msa_rssm_weights.pkl')


def download_weights(file_link, save_file_path, verbose=False):
    if not os.path.exists(rssm_weights_dir):
        os.makedirs(rssm_weights_dir)
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    r = requests.get(file_link, stream=True, headers=headers)
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    if verbose:
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                if verbose:
                    progress_bar.update(1024)
    if verbose:
        progress_bar.close()


def compare_hash(file_link, local_file_path, verbose=False):
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    r = requests.get(file_link, stream=True, headers=headers)
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    if verbose:
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    hash = hashlib.sha256()
    for data in r.iter_content(4096):
        hash.update(data)
        if verbose:
            progress_bar.update(4096)
    if verbose:
        progress_bar.close()
    online_file_hash = hash.hexdigest()

    with open(local_file_path, 'rb') as file:
        if verbose:
            print(f'opening {local_file_path}')
        hash = hashlib.sha256()
        data = file.read(4096)
        while data:
            hash.update(data)
            data = file.read(4096)
        local_file_hash = hash.hexdigest()

    return online_file_hash, local_file_hash
