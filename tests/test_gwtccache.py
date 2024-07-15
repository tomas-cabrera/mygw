import os
import os.path as pa
import shutil

from mygw.io import paths
from mygw.io.skymaps import GWTCCache


def test_gwtccache():
    """Makes a temporary cache and checks some events."""
    # Make temp cache
    cache_dir = pa.join(paths.cache_dir, "tmp")
    cache = GWTCCache(cache_dir=cache_dir)

    # Check some events
    for eventid in [
        "GW190413_134308",
        "GW190521",
        "GW190814_211039",
        "GW200322_091133",
    ]:
        path = cache.get_skymap_path_for_eventid(eventid)
        assert os.path.exists(path)

    # Remove temp cache
    shutil.rmtree(cache_dir)
