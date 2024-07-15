import glob
import os
import os.path as pa

import requests

from mygw.io import paths

gwtc2url = {
    "GWTC2": "https://dcc.ligo.org/public/0169/P2000223/007/all_skymaps.tar",
    "GWTC2.1": "https://zenodo.org/api/files/ecf41927-9275-47da-8b37-e299693fe5cb/IGWN-GWTC2p1-v2-PESkyMaps.tar.gz",
    "GWTC3": "https://zenodo.org/records/8177023/files/IGWN-GWTC3p0-v2-PESkyLocalizations.tar.gz",
}


def make_skymap_cache(cache_dir=paths.cache_dir, overwrite=False):
    for gwtc, url in gwtc2url.items():
        # Generate path
        tar_path = pa.join(cache_dir, pa.basename(url))
        dir_path = pa.join(cache_dir, gwtc)
        print(tar_path)
        print(dir_path)
        # continue

        # If does not exist or overwrite
        if not pa.exists(tar_path) or overwrite:
            # Get content
            r = requests.get(url)

            # Save content
            os.makedirs(pa.dirname(tar_path), exist_ok=True)
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # If does not exist or overwrite
        if not pa.exists(dir_path) or overwrite:
            # Extract content
            os.makedirs(dir_path, exist_ok=True)
            os.system(f"tar -xf {tar_path} -C {dir_path}")

            # Move files as needed
            # GWTC2 + GWTC3 have an intermediate directory
            if gwtc in ["GWTC2", "GWTC3"]:
                # Assume the only file is the intermediate directory
                intermediate_dir = glob.glob(f"{dir_path}/*")[0]
                os.system(f"mv {intermediate_dir}/* {dir_path}")
                os.rmdir(intermediate_dir)
