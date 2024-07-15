import glob
import os
import os.path as pa
from ligo.skymap.io.fits import read_sky_map

import requests

from mygw.io import paths


class GWTCCache:
    def __init__(
        self,
        gwtc2url={
            "GWTC2": "https://dcc.ligo.org/public/0169/P2000223/007/all_skymaps.tar",
            "GWTC2.1": "https://zenodo.org/records/6513631/files/IGWN-GWTC2p1-v2-PESkyMaps.tar.gz",
            "GWTC3": "https://zenodo.org/records/8177023/files/IGWN-GWTC3p0-v2-PESkyLocalizations.tar.gz",
        },
        cache_dir=paths.cache_dir,
        ensure_cache_on_init=True,
    ):
        # Save params
        self.cache_dir = cache_dir
        self.gwtc2url = gwtc2url

        # Make cache
        if ensure_cache_on_init:
            self.make_gwtc_cache()

    def make_gwtc_cache(self, overwrite=False):
        for gwtc, url in self.gwtc2url.items():
            # Generate path
            tar_path = pa.join(self.cache_dir, pa.basename(url))
            dir_path = pa.join(self.cache_dir, gwtc)

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

    def get_skymap_path_for_eventid(
        self,
        eventid,
        gwtc_hierarchy=["GWTC3", "GWTC2.1", "GWTC2"],
        waveform_hierarchy=["C01:NRSur7dq4", "C01:IMRPhenomXPHM", "C01:SEOBNFv4PHM"],
    ):
        # Iterate through GWTC hierarchy
        for gwtc in gwtc_hierarchy:
            # Iterate through waveform hierarchy
            for waveform in waveform_hierarchy:
                # Assemble path
                path = self.assemble_skymap_path(eventid, gwtc, waveform)

                # If exists, return
                if pa.exists(path):
                    return path

        # If not found, raise FileNotFoundError
        raise FileNotFoundError(f"Skymap not found for eventid {eventid}")

    def assemble_skymap_path(self, eventid, gwtc, waveform):
        # GWTC2 has a different naming convention
        if gwtc == "GWTC2":
            path = pa.join(self.cache_dir, gwtc, f"{eventid}_{waveform}.fits")
        # Others
        else:
            # Convert release version
            if "." not in gwtc:
                release = f"{gwtc}p0"
            else:
                release = gwtc.replace(".", "p")

            # Assemble path
            path = pa.join(
                self.cache_dir,
                gwtc,
                f"IGWN-{release}-v2-{eventid}_PEDataRelease_cosmo_reweight_{waveform}.fits",
            )

        return path


class Skymap:
    def __init__(self, filename, nest=False, distances=False, moc=False):
        # Pass args, kwargs
        self.filename = filename
        self.nest = nest
        self.distances = distances
        self.moc = moc

        # Read skymap
        self.skymap = read_sky_map(
            filename,
            nest=nest,
            distances=distances,
            moc=moc,
        )
