import glob
import math
import os
import os.path as pa
import shutil
from copy import copy

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import ligo.skymap.distance as lsm_dist
import ligo.skymap.moc as lsm_moc
import numpy as np
import requests
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable
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
        nest=True,
        distances=True,
        moc=False,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    ):
        # Pass args, kwargs
        self.filename = filename
        self.nest = nest
        self.distances = distances
        self.moc = moc
        self.cosmo = cosmo

        # Check if MOC and nest
        if self.moc and not self.nest:
            raise ValueError("MOC skymaps must be nested")

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

    def flatten(self, level):
        """Flatten the skymap to a given level."""
        # Make copy
        skymap_temp = copy(self)

        # If already flat
        if not self.moc:
            # If already at level, return self
            if self.nside == hp.order2nside(level):
                return self
            # If not, define UNIQ column
            # BUG: This doesn't handle RING-ordered maps
            skymap_temp.skymap["UNIQ"] = ah.level_ipix_to_uniq(
                level, np.arange(len(self.skymap))
            )

        # Flatten the skymap
        skymap_temp.skymap = QTable(lsm_moc.rasterize(skymap_temp.skymap, order=level))
        skymap_temp.skymap.meta = self.skymap.meta

        # Update moc, nside
        skymap_temp.moc = False
        skymap_temp.nside = hp.npix2nside(len(skymap_temp.skymap))

        # Add PROB column
        skymap_temp.skymap["PROB"] = skymap_temp.skymap[
            "PROBDENSITY"
        ] * hp.nside2pixarea(skymap_temp.nside)

        return skymap_temp

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
                moc_level_max = ah.uniq_to_level_ipix(np.max(self.skymap["UNIQ"]))[0]

            # Initialize levels
            levels = np.arange(moc_level_max + 1)
            nsides = [hp.order2nside(l) for l in levels]

            # Get uniqs for nsides, skycoord
            ipix = ah.lonlat_to_healpix(
                skycoord.ra,
                skycoord.dec,
                nsides,
                order=("nested" if self.nest else "ring"),
            )
            uniqs = ah.level_ipix_to_uniq(
                levels,
                ipix,
            )

            # Select first uniq that is in the skymap
            mask = np.isin(uniqs, self.skymap["UNIQ"])
            if np.any(mask):
                return uniqs[mask][0]
            else:
                raise ValueError(f"Skycoord {skycoord} not found in skymap")

        # If flat skymap, convert skycoord to ipix using self.nsides
        else:
            # Calculate ipix
            ipix = ah.lonlat_to_healpix(
                skycoord.ra,
                skycoord.dec,
                self.nside,
                order=("nested" if self.nest else "ring"),
            )

            return ipix

    def get_hpx_inds(self, hpxs):
        # Cast as array
        hpxs = np.array([hpxs]).flatten()

        # MOC skymaps: find index of UNIQ
        if self.moc:
            ind_hpx = [np.where(self.skymap["UNIQ"] == id)[0] for id in hpxs]
            return np.array(ind_hpx).flatten()
        # Flat skymaps: assume hpx is the index
        else:
            return hpxs

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
            dp_dOmega = self.skymap[ind_hpx]["PROBDENSITY"].value / u.sr
        else:
            dp_dOmega = self.skymap[ind_hpx]["PROB"].value / (
                hp.nside2pixarea(self.nside) * u.sr
            )

        return dp_dOmega

    def hpx_prob(self, hpx=None):
        """Returns probabilty contained in the given HEALPixs.

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

        # Fetch HEALPix indices probability
        if self.moc:
            hpx_areas = lsm_moc.uniq2pixarea(self.skymap[ind_hpx]["UNIQ"].value)
            hpx_prob = self.skymap[ind_hpx]["PROBDENSITY"].value * hpx_areas
        else:
            hpx_prob = self.skymap[ind_hpx]["PROB"].value

        return hpx_prob

    def ddL_dz_jacobian(self, z, cosmo=None):
        """Jacobian of luminosity distance with respect to redshift.

        dL(z) = (1+z) d_comoving(z); d_comoving(z) = \int_0^z dz' c/H(z')
        --> ddL/dz(z) = d_comoving(z) + (1+z) * c/H(z)

        """
        # Set cosmo
        cosmo = self.get_cosmo(cosmo)

        # Calculate jacobian
        ddL_dz_jacobian = cosmo.comoving_distance(z) + (1 + z) * c / cosmo.H(z)

        return ddL_dz_jacobian.to(u.Mpc)

    def _calc_distnorm(self, m, s, n):
        """Analytic from of the integral of the conditional pdf from 0 to infinity.

        Parameters
        ----------
        m : _type_
            DISTMU
        s : _type_
            DISTSIGMA
        n : _type_
            DISTNORM

        Returns
        -------
        _type_
            _description_
        """

        _distnorm = (
            (
                (m * s**2 * math.exp(-0.5 * (m / s) ** 2))
                + (
                    (math.pi / 2) ** 0.5
                    * s
                    * (m**2 + s**2)
                    * (1 + math.erf(0.5**0.5 * m / s))
                )
            )
            / (2 * math.pi * s**2) ** 0.5
        ) ** -1

        # Check if negative
        if _distnorm < 0:
            print(
                f"WARNING: _calc_distnorm < 0 (DISTMU={m}, DISTSIGMA={s}); returning original DISTNORM={n}"
            )
            _distnorm = n

        # Check if nan
        if np.isnan(_distnorm):
            print(
                f"WARNING: _calc_distnorm is NaN; (DISTMU={m}, DISTSIGMA={s}); returning original DISTNORM={n}"
            )
            _distnorm = n

        return _distnorm

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
        skymap_temp = self.skymap[self.get_hpx_inds(hpx)]

        # From https://iopscience.iop.org/article/10.3847/0067-0049/226/1/10:
        # """In pixels on the sky that contain very little probability,
        # sometimes the conditional distance distribution cannot be represented using the ansatz.
        # This is signaled by DISTMU = , DISTSIGMA = 1, and DISTNORM = 0."""
        # Currently we handle this by returning dp_ddL for the most probable hpx
        if (
            not np.isfinite(skymap_temp["DISTMU"])
            and skymap_temp["DISTSIGMA"] == 1
            and skymap_temp["DISTNORM"] == 0
        ):
            print(
                "WARNING: Distance ansatz failed as described in S2 of the 'Going the Distance' supplement paper",
                f"(DISTMU={skymap_temp['DISTMU'].value}, DISTSIGMA={skymap_temp['DISTSIGMA'].value}, DISTNORM={skymap_temp['DISTNORM'].value})",
                "returning value for the most probable hpx.",
            )
            if self.moc:
                probkey = "PROBDENSITY"
            else:
                probkey = "PROB"
            skymap_temp = self.skymap[[np.argmax(self.skymap[probkey])]]
        # If only DISTMU is infinite, return dL^2 distribution
        elif not np.isfinite(skymap_temp["DISTMU"]):
            print(
                "WARNING: DISTMU is infinite; returning distribution for most probable hpx",
            )
            if self.moc:
                probkey = "PROBDENSITY"
            else:
                probkey = "PROB"
            skymap_temp = self.skymap[[np.argmax(self.skymap[probkey])]]

        # Calculate dp_ddL
        dp_ddL = (
            lsm_dist.conditional_pdf(
                dL.to(u.Mpc).value,
                skymap_temp["DISTMU"].value,
                skymap_temp["DISTSIGMA"].value,
                skymap_temp["DISTNORM"].value,
            )
            / u.Mpc
        )

        # Recalculate DISTNORM
        _distnorm = self._calc_distnorm(
            skymap_temp["DISTMU"].value,
            skymap_temp["DISTSIGMA"].value,
            skymap_temp["DISTNORM"].value,
        )
        dp_ddL *= _distnorm / skymap_temp["DISTNORM"].value

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
        total_prob_grid = np.trapz(dp_dz_grid, z_grid)
        if total_prob_grid == 0 and not np.all(z_grid == 0):
            skymap_temp = self.skymap[self.get_hpx_inds(hpx)]
            print(
                "WARNING: dp_dz_grid probabilities summed to 0; returning uniform probabilites",
            )
            dp_dz_grid = np.ones_like(dp_dz_grid)
            total_prob_grid = np.trapz(dp_dz_grid, z_grid)
            dp_dz_evaluate = np.ones_like(dp_dz_evaluate)
        dp_dz_evaluate /= total_prob_grid

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

    def _pix2radec(self, hpx, dx=None, dy=None):
        if self.moc:
            _level, _ipix = ah.uniq_to_level_ipix(hpx)
            _nside = hp.order2nside(_level)
        else:
            _ipix = hpx
            _nside = self.nside

        return ah.healpix_to_lonlat(
            _ipix,
            _nside,
            order=("nested" if self.nest else "ring"),
            dx=dx,
            dy=dy,
        )

    def draw_random_location(
        self,
        z_grid,
        ci_area=None,
        ci_volume=None,
        rng=np.random.default_rng(12345),
    ):
        if ci_area is not None and ci_volume is not None:
            raise ValueError("Only one of ci_area or ci_volume should be specified")
        elif ci_area is not None:

            # Get hpxs for area
            hpx_cis = self.get_hpxs_for_ci_areas(ci_area)[0]

            # Randomly select hpx
            hpx_prob = self.hpx_prob(hpx=hpx_cis)
            hpx = rng.choice(hpx_cis, p=hpx_prob / hpx_prob.sum())

            # Convert hpx to ra, dec
            dx, dy = rng.uniform(0, 1, size=2)
            ra, dec = self._pix2radec(hpx, dx=dx, dy=dy)

            # Randomly select redshift
            dp_dz = self.dp_dz(z_grid, hpx=hpx)
            z = rng.choice(z_grid, p=dp_dz / dp_dz.sum())

        return [ra, dec, z]
