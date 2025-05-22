Joint Limlam Mocker: A package for Line Intensity Mocks
=================================================

This is a fork of Dongwoo Chung and George Stein's limlam mocker package, intended to create mock line intensity maps and galaxy catalogs for joint LIM analyses. It has been modified heavily from the original version. Additions include the second spectral line tracer (which can be filtered into a mock catalog), line broadening and beam smoothing, and simulated instrumental effects (such as instrumental noise or spectral line interlopers). The package is summarized in detail in `Dunne et al. 2025 <https://arxiv.org/abs/2503.21743>`_.

To Run
------

parameters are mostly very self explanatory, and are set up with argparse. You can pass a parameter file with adjusted parameters using `-p` when running, or adjust the parameters directly in the code, as is shown in `lim_mocker.py`. To run the full code, use:

        >>> ./lim_mocker.py -p ./limlam_mocker/param_example.txt


This will load in the halo catalogue, assign luminosities to each halo, and bin them up into a 3D intensity map data cube of size (npix_x, npix_y, nmaps). This map will then be saved using the `npz format <https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.savez.html>`_. This file contains all required info for the maps - field of view, pixel size, frequency of maps, etc...

A small sample halo catalogue is included, but many more of larger sizes are available by contacting the author. 

the basic workflow
------------------
The example script provided in `lim_mocker.py` shows the use of a wrapper function (`simgenerator`), but the more granular workflow is:

- `parser.parse_args` or `SimParameters` initialises the parameters for the mock map.
- `HaloCatalog` loads the halos from the lightcone specified into a halo object and implements a mass and field of view cutoff.
- `Mhalo_to_Ls` calculates luminosities for all halos in both tracers (calling `Mhalo_to_Lco` for the CO luminosities and `Mhalo_to_Lcatalog` for the other tracer luminosities).
- `halos.get_velocities` generates the halo velocities used for astrophysical line broadening in the CO halos.
- `SimMap` generates a map object from given parameters.
- `map.mockmapmaker`  populates this map with flux (units of T) from the halos.
- `map.write` saves the maps to the npz file as described above, and `halos.write_cat` saves the halo luminosities.
- `map_to_pspec` then also calculates a 3D spherically averaged power spectrum for this map.

To add your own L_CO(M,z,...) function
--------------------------------------
add it to `halos_to_luminosity.py` (under `Mhalo_to_Lco`), following the ones already there eg:    

        >>> dict = {'Li':          Mhalo_to_Lco_Li,
        >>>        'Padmanabhan': Mhalo_to_Lco_Padmanabhan}
            
        >>> def Mhalo_to_Lco_Li(halos, coeffs):
        >>>        ...

For testing, you could also make use of the 'arbitrary' model, but this is strongly discouraged, and adding new prescriptions to halos_to_luminosity.py is strongly encouraged.

To add your own L_cat(M,z,...) function 
---------------------------------------
add it to halos_to_luminosity.py (under `Mhalo_to_Lcatalog`), following the ones already there eg:

        >>> dict = {'schechter':          Mhalo_to_Lcatalog_schechter,
        >>>        'newlya':              Mhalo_to_Lcatalog_newlya}
            
        >>> def Mhalo_to_Lcatalog_newlya(halos, params):
        >>>        ...

To add your own halo catalogue
------------------------------
use the template in load_halos.py

Advanced usage: using `limlam_mocker` directly in your own code
---------------------------------------------------------------
This repo is a complete program in itself, designed so that you can painlessly run `lim_mocker.py` and not have to worry about the guts. If you want to use code other than `lim_mocker.py` with this library, the `limlam_mocker` folder inside this repository will function as a self-contained Python package.

Depending on the OS and the Python version, what may work is to simply place this folder in your local site-packages directory or create a symlink there. So if the contents of e.g. `~/.local/lib/python3.5/site-packages/limlam_mocker` (the local site-packages directory for Python 3.5 in Linux) are equal to the contents of the `limlam_mocker` folder in this repo (not the root contents of this repo!), you should be able to take advantage of `limlam_mocker` functions from any Python script on your computer, regardless of where it is placed.

Extended usage: using modules from `limlam_mocker.extensions` in your code
--------------------------------------------------------------------------
The basic `limlam_mocker` module allows for calculation of line-intensity auto-correlation spectra. To work with uncertainties on this power spectrum or scenarios like cross-correlation with galaxy surveys (see arXiv:1809.04550), you can also do this:
        >>> from limlam_mocker.extensions import llm_xcorr as llmx
        >>> from limlam_mocker.extensions import llm_error as llme
`llm_xcorr` provides an extended version of the default `map_to_pspec.py` code, reducing redundant definitions of k-space parameters and allowing calculations of auto and cross spectra. `llm_error` automatically allows for calculation of noise power spectra for heterodyne receivers and all-k signal-to-noise for auto and cross spectra. Documentation on both is currently sparse, but should be improved in future.

This code was written by George Stein    - gstein@cita.utoronto.ca, with many additions by Dongwoo Chung - dongwooc@stanford.edu. It was then cannibalized and reformatted (with some additions) by Delaney Dunne - ddunne@astro.caltech.edu.

A version of the original (non-joint) code with many more options and functions (useful power spectrum calculation too!) can be found at https://github.com/dongwooc/imapper2, written by Tony Li and Dongwoo Chung.

License
-------

limlam_mocker is licensed under the GNU General Public License v3.0 
