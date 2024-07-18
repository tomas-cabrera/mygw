import glob
import os
import os.path as pa

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import ligo.skymap.distance as lsm_dist
import numpy as np
import requests
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.io.fits import read_sky_map

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
    def __init__(
        self,
        filename,
        nest=False,
        distances=False,
        moc=False,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    ):
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

        # If flat skymap, set nside
        if not moc:
            self.nside = hp.get_nside(self.skymap)

    def get_cosmo(self, cosmo):
        """Returns default cosmo if None, else returns input."""
        if cosmo is None:
            return self.cosmo
        return cosmo

    def skycoord2pix(self, skycoord, moc_level_max=None):
        """Converts skycoord to HEALPix pixel."""
        # If multiorder, iterate through levels until a match is found
        if self.moc:
            # If moc_level_max is None, set to max level in skymap
            if moc_level_max is None:
                moc_level_max = hp.uniq_to_level_ipix(np.max(self.skymap["UNIQ"]))[0]

            # Initialize levels
            levels = np.arange(moc_level_max)

            # Get uniqs for nsides, skycoord
            uniqs = ah.level_ipix_to_uniq(
                levels,
                hp.ang2pix(
                    [hp.order2nside(l) for l in levels],
                    skycoord.ra.deg,
                    skycoord.dec.deg,
                    lonlat=True,
                    nest=self.nest,
                ),
            )

            # Select first uniq that is in the skymap
            mask = np.isin(uniqs, self.skymap["UNIQ"])
            if mask.any():
                return self.skymap["UNIQ"][mask][0]
            else:
                raise ValueError(f"Skycoord {skycoord} not found in skymap")

        # If flat skymap, convert skycoord to ipix using self.nsides
        else:
            # Calculate ipix
            ipix = hp.ang2pix(
                self.nside,
                skycoord.ra.deg,
                skycoord.dec.deg,
                lonlat=True,
                nest=self.nest,
            )

            return ipix

    def get_hpx_inds(self, id_hpx):
        # MOC skymaps: find index of UNIQ
        if self.moc:
            ind_hpx = [np.where(self.skymap["UNIQ"] == id)[0] for id in id_hpx]
            return np.array(ind_hpx).flatten()
        # Flat skymaps: assume id_hpx is the index
        else:
            return id_hpx

    def dp_dOmega(self, id_hpx):
        """Returns probabilty density over solid angle for the given HEALPixs.

        Parameters
        ----------
        id_hpx : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Get indices
        ind_hpx = self.get_hpx_inds(id_hpx)

        # Fetch HEALPix indices probdensity
        if self.moc:
            dp_dOmega = self.skymap[ind_hpx]["PROBDENSITY"] / u.sr
        else:
            dp_dOmega = self.skymap[ind_hpx]["PROB"] / (
                hp.nside2pixarea(self.nside) * u.sr
            )

        return dp_dOmega

    def ddL_dz_jacobian(self, z, cosmo=None):
        """Jacobian of luminosity distance with respect to redshift.

        dL(z) = (1+z) d_comoving(z); d_comoving(z) = \int_0^z dz' c/H(z')
        --> ddL/dz(z) = d_comoving(z) + (1+z) * c/H(z)

        """
        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Calculate jacobian
        ddL_dz_jacobian = cosmo.comoving_distance(z) + (1 + z) * u.c / cosmo.H(z)

        return ddL_dz_jacobian.to(u.Mpc)

    def dp_ddL(self, dL, id_hpx):
        """_summary_

        Parameters
        ----------
        dL : _type_
            _description_
        id_hpx : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Select hpx
        skymap_temp = self.skymap[id_hpx]

        # Calculate dp_ddL
        dp_ddL = lsm_dist.conditional_pdf(
            dL,
            skymap_temp["DISTMU"],
            skymap_temp["DISTSIGMA"],
            skymap_temp["DISTNORM"],
        )

        return dp_ddL

    def dp_dz(
        self,
        z,
        id_hpx,
        cosmo=None,
    ):
        """Returns the probability density over redshift, over the domain of the skymap.

        Parameters
        ----------
        z : _type_
            _description_
        cosmo : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Calculate dp_ddL
        dp_ddL = self.dp_ddL(
            cosmo.luminosity_distance(z),
            id_hpx,
        )

        # Convert to dp_dz
        dp_dz = dp_ddL * self.ddL_dz_jacobian(z, cosmo=cosmo)

        return dp_dz

    def dp_dz_grid(
        self,
        z_grid,
        id_hpx,
        z_evaluate=None,
        cosmo=None,
    ):
        """Returns the probability density over redshift, over the specified redshift domain.

        Parameters
        ----------
        z_grid : _type_
            _description_
        z_evaluate : _type_, optional
            _description_, by default None
        cosmo : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # If z_evaluate is None, set to z_grid
        if z_evaluate is None:
            z_evaluate = z_grid

        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Calculate dp_dz over z_evaluate
        dp_dz_evaluate = self.dp_dz(z_evaluate, id_hpx, cosmo=cosmo)

        # Calculate dp_dz over z_grid
        dp_dz_grid = self.dp_dz(z_grid, id_hpx, cosmo=cosmo)

        # Normalize dp_dz_evalute
        dp_dz_evaluate /= dp_dz_grid.sum()

        return dp_dz_evaluate
