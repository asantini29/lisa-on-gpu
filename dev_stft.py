from functools import partial

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass

from scipy.signal.windows import hann, tukey

from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import DefaultOrbits
from lisatools.utils.constants import *

from lisatools.domains import WAVELET_DURATION, TDSignal, TDSettings, FDSignal, FDSettings, get_stft_settings
from lisatools.domaincomputation import STFTComputationGroup
from fastlisaresponse.gbcomps import STFTGBComputations

import matplotlib.pyplot as plt
DOPLOT = False
TUKEY_ALPHA = 0.1
DECIMATE = 100

if __name__ == "__main__":

    force_backend = "cpu"
    gpus = [0] if force_backend == "gpu" else None

    xp = np if force_backend == "cpu" else cp
    orbits = DefaultOrbits(force_backend=force_backend)
    orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
    t0 = 0.0
    dt = 10.0
    stft_dt = 6 * 3600.0
    Tobs = 2 * YRSID_SI
    Tobs = stft_dt * int(Tobs / stft_dt)
    t_ref = 0.0
    Nobs = int(Tobs / dt)
    N_sparse = 256
    t_tdi = xp.linspace(0.0, Tobs, N_sparse + 1)[1:-1]
    Tobs = Nobs * dt

    num_bin = 3

    data_t_arr = xp.arange(Nobs) * dt
    stft_settings = get_stft_settings(data_t_arr, stft_dt, min_freq=1e-5, force_backend=force_backend)

    keep = (data_t_arr > t_tdi[0]) & (data_t_arr < t_tdi [-1])
    tdi_t_arr = data_t_arr[keep]

    amp = xp.full(num_bin, 1e-23)
    f0 = xp.full(num_bin, 4.2300812341e-3)
    fdot = xp.full(num_bin, 1e-18)
    fddot = xp.full(num_bin, 0.0)
    phi0 = xp.full(num_bin, 0.892342342342)
    inc = xp.full(num_bin, 1.2309804223)
    psi = xp.full(num_bin, 3.00908098)
    lam = xp.full(num_bin, 4.827342308)
    beta = xp.full(num_bin, -0.50923423)

    gb_gen = GBTDIonTheFly(
        t_tdi, 
        Tobs,
        t_ref,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )

    output = gb_gen(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, convert_to_ra_dec=False, return_spline=True)
    tdi_output = xp.zeros((num_bin, 3, len(data_t_arr))) 

    tdi_output[:, :, keep]= output.eval_tdi(tdi_t_arr)
    
    print('TDI output generated')

    if DOPLOT:
        plt.plot(data_t_arr[::DECIMATE], tdi_output[0,0, ::DECIMATE])
        plt.show()

    nperseg = stft_settings.get_nperseg(dt)
    td_signal = TDSignal(tdi_output[0], settings=TDSettings(t_ref, tdi_output.shape[-1], dt, force_backend=force_backend))
    
    window_fn = partial(tukey, alpha=TUKEY_ALPHA)
    #window_fn = hann
    window = xp.array(window_fn(nperseg), dtype=xp.float64)
    window_factor = xp.sum(window) / nperseg

    stft_signal = td_signal.stft(window=window, settings=stft_settings)

    from lisatools.datacontainer import DataResidualArray
    from lisatools.sensitivity import XYZSensitivityBackend
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from copy import deepcopy
    data_res = DataResidualArray(stft_signal)


    sens_mat = XYZSensitivityBackend(orbits=orbits, settings=stft_settings, force_backend=force_backend)
    Soms = 15e-12
    Sa =  3e-15
    freqs = stft_settings.f_arr

    matrix = sens_mat.compute_sensitivity_matrix(sens_mat.basis_settings.f_arr, Soms, Sa)
    matrix.shape

    sens_mat.sens_mat = matrix

    num_container = 5
    acs_list = []
    for i in range(num_container):
        analysis_container = AnalysisContainer(deepcopy(data_res), deepcopy(sens_mat))
        acs_list.append(analysis_container)
    
    acs = AnalysisContainerArray(acs_list, gpus=gpus)
    check_ll = acs.likelihood(source_only=True)

    print('Analysis container array created')

    stft_group = STFTComputationGroup(acs, split_index=0, window_alpha=TUKEY_ALPHA, force_backend=force_backend)

    stft_group.compute_d_d_term()
    print('d_d term computed')
    print(f"stft_group.d_d: {stft_group.d_d}")

    gb_comps = STFTGBComputations(
        stft_comps=stft_group,
        T=Tobs,
        t_ref=t_ref,
        orbits=orbits,
        tdi_config=tdi_config,
        force_backend=force_backend,
        n_side_bins=5,
    )

    params = xp.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

    test = gb_comps.get_ll_stft(params, data_index=None, noise_index=None)

    print(f"LL: {test}")

    n_bins = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 200, 500]
    like_out = []
    for n_bin in n_bins:
        gb_comps.n_side_bins = n_bin
        like_out.append(gb_comps.get_ll_stft(params, data_index=None, noise_index=None)[0])

    title = "With window effect" if stft_group.window_alpha > 0 else "Without window effect"
    
    if gb_comps.window_factor != 1.0:
        title += f", window factor = {gb_comps.window_factor:.3f}"

    plt.figure(); plt.plot(n_bins, like_out); plt.xlabel('number of side bins'); plt.ylabel('logl'); plt.title(title); plt.show()

    breakpoint()