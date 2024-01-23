import numpy              as np
import os # for figure saving
import matplotlib.pylab   as plt
import scipy              as sp
import limlam_mocker      as llm
from limlam_mocker.extensions import llm_doppler as llmd
import csv

import lparams_fiuducial_allfields_broadened as params

llm.debug.verbose = True
llm.write_time('Starting Line Intensity Mapper')

for i in range(len(params.halo_catalogue_files)):
    params.halo_catalogue_file = params.halo_catalogue_files[i]

    # set up the output files to save to
    llm.make_output_filenames(params, outputdir=params.output_dir)

    ### setup maps to output
    mapinst = llm.params_to_mapinst(params)

    ### load halos from catalogue
    halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file, output=params.halo_output_file)
    halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst)

    if params.broaden:

        halos = llmd.get_halo_velocities(halos, cosmo, params)

        ### Calculate Luminosity of each halo
        halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs)

        ### If a mass cutoff is input, split into low- and high-mass halos before
        ### mapmaking
        if params.mass_cutoff:
            halos_loM = llmd.halo_masscut_subset(halos, params.min_mass, params.mass_cutoff)
            halos_loM.output_file = halos_loM.output_file + '_lowmass'
            halos_hiM = llmd.halo_masscut_subset(halos, params.mass_cutoff, 1e20)
            halos_hiM.output_file = halos_hiM.output_file + '_highmass'


            mapinst_loM = mapinst.copy()
            mapinst_loM.output_file = mapinst_loM.output_file + '_lowmass'

            mapinst_hiM = mapinst.copy()
            mapinst_hiM.output_file = mapinst_hiM.output_file + '_highmass'

            mapinst_loM.maps = llmd.Lco_to_map_doppler_synthbeam(halos_loM, mapinst_loM,
                                                            bincount=params.velocity_bins,
                                                            binattr=params.freq_attr,
                                                            fwhmattr=params.freq_attr,
                                                            freqrefine=params.freq_subpix,
                                                            xrefine=params.x_subpix,
                                                            beamfwhm=params.beam_fwhm)
            mapinst_hiM.maps = llmd.Lco_to_map_doppler_synthbeam(halos_hiM, mapinst_hiM,
                                                            bincount=params.velocity_bins,
                                                            binattr=params.freq_attr,
                                                            fwhmattr=params.freq_attr,
                                                            freqrefine=params.freq_subpix,
                                                            xrefine=params.x_subpix,
                                                            beamfwhm=params.beam_fwhm)

            # save maps
            llm.save_maps(mapinst_loM)
            llm.save_maps(mapinst_hiM)

            # save the cut catalogue
            llm.save_halos(halos_loM, trim=5000)
            llm.save_halos(halos_hiM, trim=5000)


        ### Bin luminosities into map, applying the broadening spectrally and spatially
        ### this is the version without any cutoffs at all -- identical to adding the
        ### high- and low-mass maps together
        mapinst.maps = llmd.Lco_to_map_doppler_synthbeam(halos, mapinst,
                                                        bincount=params.velocity_bins,
                                                        binattr=params.freq_attr,
                                                        fwhmattr=params.freq_attr,
                                                        freqrefine=params.freq_subpix,
                                                        xrefine=params.x_subpix,
                                                        beamfwhm=params.beam_fwhm)
        # save this output
        llm.save_maps(mapinst)
        llm.save_halos(halos, trim=5000)

        # end if params.mass_cutoff
    # end if params.broaden
    else:
        # if no broadening, just do the normal map run
        halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst)

        ### Calculate Luminosity of each halo
        halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs)

        ### Bin halo luminosities into map
        mapinst.maps = llm.Lco_to_map(halos,mapinst)

        ### Output map to file
        llm.save_maps(mapinst)
        llm.save_halos(halos, trim=5000)

        ### Calculate power spectrum
        k,Pk,Nmodes = llm.map_to_pspec(mapinst,cosmo)
        Pk_sampleerr = Pk/np.sqrt(Nmodes)

        ### Plot results
        llm.plot_results(mapinst,k,Pk,Pk_sampleerr,params)

        llm.write_time('Finished Line Intensity Mapper')

# end for file in halofiles
