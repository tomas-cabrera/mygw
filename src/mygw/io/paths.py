import os
import os.path as pa

import mygw

# Path to module
mygw_dir = mygw.__file__

# Path to data
data_dir = pa.join(pa.dirname(mygw_dir), "data")

# Path to cache
cache_dir = pa.join(data_dir, ".cache")
