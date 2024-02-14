from __future__ import annotations
import numpy as np
import scipy.interpolate as interp
import numpy.typing as ntyping
import sys

from dataclasses import dataclass, field
from pixell import enmap, utils
import os

from .load_halos import *
from .halos_to_luminosity import *
from .luminosity_to_map import *

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)


def simgenerator(params):
    """
    wrapper function to run the simulation
    """

    # generate the halo catalog from the passed peak-patch catalog
    halos = HaloCatalog(params, params.halo_catalog_file)

    # generate luminosities for each halo
    Mhalo_to_Ls(halos, params)

    # generate simulation velocities if required
    if params.freqbroaden:
        halos.get_velocities(params)

    # set up map parameters
    map = SimMap(params)

    # make the map
    map.mockmapmaker(halos, params)

    # cut catalog based on observability
    halos.attrcut_subset('Lcat', params.lcat_cutoff, np.nanmax(halos.Lcat)+10, params, in_place=True)

    # output files for map and catalog
    params.map_output_file = params.output_dir + '/sim_map.npz'
    params.cat_output_file = params.output_dir + '/sim_cat.npz'

    map.write(params)
    halos.write_cat(params)

    return