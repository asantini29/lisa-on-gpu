from __future__ import annotations
from .utils.parallelbase import FastLISAResponseParallelModule
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import Orbits, EqualArmlengthOrbits
from copy import deepcopy
from lisatools.domains import WDMLookupTable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp
    
    from lisatools.domaincomputation import STFTComputationGroup

import numpy as np


class GBComputations(FastLISAResponseParallelModule):
    def __init__(self, 
                 T: float, 
                 t_ref: float = 0.0,
                 orbits: Orbits = None, 
                 tdi_config: str | TDIConfig = None, 
                 force_backend: str = None, 
                 ):
        
        super().__init__(force_backend=force_backend)
        # setup orbits
        self.orbits = orbits
         # setup TDI info
        self.tdi_config = tdi_config
        self.T = T

        # GB generator reference time
        self.t_ref = t_ref

    @property
    def T(self) -> float:
        """Return observation time in seconds."""
        return self._T
    @T.setter
    def T(self, T: float) -> None:
        """Set observation time."""
        self._T = T

    @property
    def t_ref(self) -> float:
        """Return reference time for GB generator."""
        return self._t_ref
    @t_ref.setter    
    def t_ref(self, t_ref: float) -> None:
        """Set reference time for GB generator."""
        self._t_ref = t_ref

    @property
    def num_params(self) -> int:
        """Return number of parameters for the GB model."""
        return 9
        
    @property
    def tdi_config(self) -> TDIConfig:
        return self._tdi_config
    
    @tdi_config.setter
    def tdi_config(self, tdi_config: str | TDIConfig):
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation")
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config)
        elif not isinstance(tdi_config, TDIConfig):
            raise ValueError("TDI Config needs to be a string or an instnace of TDIConfig.")
        self._tdi_config = tdi_config

        self.cpp_tdi_config = self.backend.TDIConfigWrap(*self._tdi_config.pytdiconfig_args)
       
    @property
    def xp(self) -> object:
        return self.backend.xp
    
    @property
    def orbits(self) -> object:
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        """Set response orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()
        
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            # assumed default arguments if not initialized as input
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)

        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)

        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

class GBWDMComputations(GBComputations):
    def __init__(self, wdm_lookup_table, T, orbits=None, tdi_config=None, force_backend=None, d_d=0.0):
        
        super().__init__(T=T, orbits=orbits, tdi_config=tdi_config, force_backend=force_backend, d_d=d_d)
        self.wdm_lookup_table = wdm_lookup_table
        self.d_d = d_d

    @property
    def wdm_lookup_table(self) -> object:
        return self._wdm_lookup_table

    @wdm_lookup_table.setter
    def wdm_lookup_table(self, wdm_lookup_table: WDMLookupTable) -> None:
        """Set wdm lookup table."""

        self._wdm_lookup_table = wdm_lookup_table
        self.c_nm_all = self.xp.asarray(wdm_lookup_table.table_sin.copy())
        self.s_nm_all = self.xp.asarray(wdm_lookup_table.table_cos.copy())

        self.cpp_wdm_lookup_table = self.backend.WaveletLookupTableWrap(
            self.c_nm_all, 
            self.s_nm_all, 
            wdm_lookup_table.f_steps, 
            wdm_lookup_table.fdot_steps, 
            wdm_lookup_table.delta_f,  # NOT .df (that is the WDM basis info) 
            wdm_lookup_table.delta_fdot, 
            wdm_lookup_table.min_f_scaled,
            wdm_lookup_table.min_fdot,
            wdm_lookup_table.df,
            wdm_lookup_table.dt,
            wdm_lookup_table.NF,
            wdm_lookup_table.NT,
            wdm_lookup_table.num_channel
        )

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None):
        params_tmp = self.xp.atleast_2d(self.xp.asarray(params))
        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        self.d_h_out = self.xp.zeros(num_bin)
        self.h_h_out = self.xp.zeros(num_bin)

        # TODO: move this part
        # TODO: need to check for num_data, num_noise
        num_data = num_noise = len(wdm_holder)
        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.df, 
            self.wdm_lookup_table.dt,
            self.wdm_lookup_table.NF, 
            self.wdm_lookup_table.NT,
            self.tdi_config.nchannels, 
            num_data, 
            num_noise
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert data_index.dtype == self.xp.int32
            
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert noise_index.dtype == self.xp.int32
            
        nparams = 9

        breakpoint()
        self.backend.GBComputationGroupWrap().gb_wdm_get_ll(
            self.d_h_out, 
            self.h_h_out, 
            self.cpp_orbits,
            self.cpp_tdi_config, 
            self.cpp_wdm_lookup_table, 
            self.cpp_wdm, 
            params_in, 
            data_index, 
            noise_index, 
            num_bin,
            nparams, 
            self.T,
            self.backend.TDITypeDict["XYZ"]
        )

        like_out = -1. / 2. * (self.d_d + self.h_h_out - 2 * self.d_h_out)
        # TODO: phase maximize

        return like_out

    def fill_global_wdm(self, templates, params, wdm_holder, data_index=None):
        assert isinstance(templates, self.xp.ndarray)

        if templates.ndim == 1:
            num_templates = int(templates.shape[-1] / (self.wdm_lookup_table.nchannels * self.wdm_lookup_table.num_m * self.wdm_lookup_table.num_n))
            assert num_templates * self.wdm_lookup_table.nchannels * self.wdm_lookup_table.num_m * self.wdm_lookup_table.num_n == templates.shape[-1]

        elif templates.ndim == 2:
            raise ValueError("Template must be 3D (nchannels, Nf, Nt), 4D (num_templates, nchannels, Nf, Nt), or flattended to 1D.")
        elif templates.ndim == 3:
            num_templates = 1
            nchannels, _num_m, _num_n = templates.shape

        elif templates.ndim == 4:
            num_templates, nchannels, _num_m, _num_n = templates.shape
            
        assert (
            nchannels == self.wdm_lookup_table.nchannels
            and _num_m == self.wdm_lookup_table.Nf
            and _num_n == self.wdm_lookup_table.Nt
        )
        templates = templates.flatten()

        params_tmp = self.xp.atleast_2d(self.xp.asarray(params))
        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        # TODO: move this part
        # TODO: need to check for num_data, num_noise
        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.settings.layer_df, 
            self.wdm_lookup_table.settings.layer_dt,
            self.wdm_lookup_table.settings.Nf,
            self.wdm_lookup_table.settings.Nt, 
            self.tdi_config.nchannels,
            self.wdm_lookup_table.is_m_ref_n_ref_even, 
            num_templates, # datqa not needed here
            num_templates  # noise not needed here
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert data_index.dtype == self.xp.int32
            
        assert data_index.max() < num_templates
        nparams = 9

        self.backend.GBComputationGroupWrap().gb_wdm_fill_global(
            templates, 
            self.cpp_orbits,
            self.cpp_tdi_config, 
            self.cpp_wdm_lookup_table, 
            self.cpp_wdm, 
            params_in, 
            data_index, 
            num_bin,
            nparams, 
            self.T,
            self.backend.TDITypeDict["XYZ"]
        )
        breakpoint()
class STFTGBComputations(GBComputations):
    """Class for GB computations using STFT domain.
    
    """
    def __init__(self,
                 stft_comps: STFTComputationGroup,
                 T: float,
                 t_ref: float = 0.0,
                 orbits: Orbits = None,
                 tdi_config: str | TDIConfig = None,
                 force_backend: str = None,
                 n_side_bins: int = 2,
                 window_factor: float = 1.0
                ):
        super().__init__(T=T, t_ref=t_ref, orbits=orbits, tdi_config=tdi_config, force_backend=force_backend)
        self.stft_comps = stft_comps
        self.n_side_bins = n_side_bins
        self.window_factor = window_factor

    @property
    def stft_comps(self) -> STFTComputationGroup:
        return self._stft_comps

    @stft_comps.setter
    def stft_comps(self, stft_comps: STFTComputationGroup) -> None:
        self._stft_comps = stft_comps

    def get_ll_stft(self,
                    params: np.ndarray | cp.ndarray,
                    data_index=None,
                    noise_index=None,
                    phase_maximize: bool = False
                    ) -> np.ndarray | cp.ndarray:
        """Compute log-likelihood for given parameters and data/noise indices."""
        params_tmp = self.xp.atleast_2d(self.xp.asarray(params))
        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        d_h_out = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        h_h_out = self.xp.zeros(num_bin, dtype=self.xp.complex128)

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert data_index.dtype == self.xp.int32

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert noise_index.dtype == self.xp.int32

        self.backend.STFTGBComputationGroupWrap().get_ll(
            d_h_out,
            h_h_out,
            self.cpp_orbits,
            self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel,
            self.stft_comps.cpp_domain,
            params_in,
            data_index,
            noise_index,
            num_bin,
            self.num_params,
            self.T,
            self.t_ref,
            self.n_side_bins,
            self.window_factor
        )

        if phase_maximize:
            raise NotImplementedError("Phase maximization not implemented yet.")
        
        print(f"d_h_out: {d_h_out}")
        print(f"h_h_out: {h_h_out}")

        like_out = -1. / 2. * (self.stft_comps.d_d[data_index] + h_h_out - 2 * d_h_out)

        return like_out.real