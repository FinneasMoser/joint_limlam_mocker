#!/usr/bin/env python
from __future__ import division
import numpy              as np
import scipy              as sp
import limlam_mocker      as llm


# get parameters for run
params = llm.parser.parse_args()

# adjust any parameters you'd like (or pass a parameter file with -p when running)
params.halo_catalog_file = '/home/deedunne/Documents/COMAP/limlam_mocker_auxcode/peakpatch_catalogues/0/COMAP_z2.39-3.44_1140Mpc_seed_13587.npz'
params.catalog_model = 'schechter' # method used to generate the catalog (here, lya, luminosities)
params.catalog_coeffs = [0.849e43, 3.9e-4, -1.8, 39, 45]
params.output_dir = '/home/deedunne/Documents/COMAP/limlam_mocker_auxcode/tester' # where maps are saved

llm.write_time('Starting Line Intensity Mapper')

# run using wrapper function
llm.simgenerator(params)

llm.write_time('Finished Line Intensity Mapper')
