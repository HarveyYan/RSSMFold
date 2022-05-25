import os
import pathlib
import requests
import RSSMFold

single_seq_rssm_weights_link = 'https://www.dropbox.com/s/b3kavezmyq94t8j/single_seq_rssm_weights.pkl?dl=0'
cov_rssm_weights_link = 'https://www.dropbox.com/s/26uh7b81g2hc3gm/cov_rssm_weights.pkl?dl=0'

basedir = pathlib.Path(RSSMFold.__file__).parent.parent.resolve()
rssm_weights_dir = os.path.join(basedir, 'RSSMFold', 'rssm_weights')
single_seq_rssm_weights_path = os.path.join(rssm_weights_dir, 'single_seq_rssm_weights.pkl')
cov_rssm_weights_path = os.path.join(rssm_weights_dir, 'cov_rssm_weights.pkl')


def download_weights(file_link, save_file_path):
    if not os.path.exists(rssm_weights_dir):
        os.makedirs(rssm_weights_dir)
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    r = requests.get(file_link, stream=True, headers=headers)
    with open(save_file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
