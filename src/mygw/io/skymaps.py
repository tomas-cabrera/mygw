import glob
import os
import os.path as pa
import shutil

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import ligo.skymap.distance as lsm_dist
import numpy as np
import requests
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.io.fits import read_sky_map
import ligo.skymap.moc as lsm_moc

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
        """Generate cache for GWTC skymaps.
        Does this by downloading from the initialized URLs.
        This should include skymaps for all events in LVK O1-3.

        Parameters
        ----------
        overwrite : bool, optional
            _description_, by default False
        """
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

    def clear_cache(self):
        """Clears the cache."""
        shutil.rmtree(self.cache_dir)

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

    def get_hpx_inds(self, hpx):
        # MOC skymaps: find index of UNIQ
        if self.moc:
            ind_hpx = [np.where(self.skymap["UNIQ"] == id)[0] for id in hpx]
            return np.array(ind_hpx).flatten()
        # Flat skymaps: assume hpx is the index
        else:
            return hpx

    def dp_dOmega(self, hpx=None):
        """Returns probabilty density over solid angle for the given HEALPixs.

        Parameters
        ----------
        hpx : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Get indices; get all indices if None
        if hpx is None:
            ind_hpx = np.arange(len(self.skymap))
        else:
            ind_hpx = self.get_hpx_inds(hpx)

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

    def dp_ddL(self, dL, hpx):
        """_summary_

        Parameters
        ----------
        dL : _type_
            _description_
        hpx : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Select hpx
        skymap_temp = self.skymap[hpx]

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
        hpx,
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
            hpx,
        )

        # Convert to dp_dz
        dp_dz = dp_ddL * self.ddL_dz_jacobian(z, cosmo=cosmo)

        return dp_dz

    def dp_dz_grid(
        self,
        z_grid,
        hpx,
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
        dp_dz_evaluate = self.dp_dz(z_evaluate, hpx, cosmo=cosmo)

        # Calculate dp_dz over z_grid
        dp_dz_grid = self.dp_dz(z_grid, hpx, cosmo=cosmo)

        # Normalize dp_dz_evalute
        dp_dz_evaluate /= dp_dz_grid.sum()

        return dp_dz_evaluate

    def get_hpxs_for_ci_areas(self, ci_areas):
        # Preprocess so that ci_areas is an array
        ci_areas = np.array([ci_areas]).flatten()

        # Get hpx probabilty array
        if self.moc:
            hpxs = self.skymap["UNIQ"]
            probs = (
                self.skymap["PROBDENSITY"]
                * lsm_moc.uniq2pixarea(self.skymap["UNIQ"])
                * u.sr
            )
            prob_densities = self.dp_dOmega()
        else:
            hpxs = np.arange(len(self.skymap))
            probs = self.skymap["PROB"]
            prob_densities = self.dp_dOmega()

        # Sort by probdensity
        sort_inds = np.argsort(prob_densities)[::-1]

        # Cumulative sum
        cumsum_probs = np.cumsum(probs[sort_inds])

        # Get cutoff
        ind_ci_cutoffs = np.searchsorted(cumsum_probs, ci_areas)

        # Get hpxs
        hpx_cis = [hpxs[sort_inds[:i]] for i in ind_ci_cutoffs]

        return hpx_cis

    def draw_random_location(
        self,
        z_grid,
        ci_area=None,
        ci_volume=None,
        rng_np=np.random.default_rng(12345),
    ):
        if ci_area is not None and ci_volume is not None:
            raise ValueError("Only one of ci_area or ci_volume should be specified")
        elif ci_area is not None:

            # Get hpxs for area
            hpx_cis = self.get_hpxs_for_ci_areas(ci_area)

            # Randomly select hpx, convert to ra, dec
            hpx = rng_np.choice(hpx_cis, p=self.dp_dOmega(hpx=hpx_cis))
            ra, dec = hp.pix2ang(self.nside, hpx, lonlat=True, nest=self.nest) * u.deg

            # Randomly select redshift
            z = rng_np.choice(z_grid, p=self.dp_dz_grid(z_grid, hpx=hpx))

        return ra, dec, z
