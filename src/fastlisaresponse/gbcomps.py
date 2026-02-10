from .utils.parallelbase import FastLISAResponseParallelModule
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import Orbits, EqualArmlengthOrbits
from copy import deepcopy
from lisatools.domains import WDMLookupTable


class GBWDMComputations(FastLISAResponseParallelModule):
    def __init__(self, wdm_lookup_table, T, orbits=None, tdi_config=None, force_backend=None):
        
        super().__init__(force_backend=force_backend)
        # setup orbits
        self.orbits = orbits
         # setup TDI info
        self.tdi_config = tdi_config
        # setup WDM c class
        self.wdm_lookup_table = wdm_lookup_table
        self.T = T
        
    @property
    def tdi_config(self) -> TDIConfig:
        return self._tdi_config
    
    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig):
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

    @property
    def wdm_lookup_table(self) -> object:
        return self._wdm_lookup_table

    @wdm_lookup_table.setter
    def wdm_lookup_table(self, wdm_lookup_table: WDMLookupTable) -> None:
        """Set wdm lookup table."""

        self._wdm_lookup_table = wdm_lookup_table
        self.c_nm_all = wdm_lookup_table.table.real.copy()
        self.s_nm_all = wdm_lookup_table.table.imag.copy()
        differential_component = wdm_lookup_table.differential_component if hasattr(wdm_lookup_table, "differential_component") else 1.0
        self.cpp_wdm_lookup_table = self.backend.WaveletLookupTableWrap(
            self.c_nm_all, 
            self.s_nm_all, 
            wdm_lookup_table.f_steps, 
            wdm_lookup_table.fdot_steps, 
            wdm_lookup_table.deltaf,  # NOT .df (that is the WDM basis info) 
            wdm_lookup_table.d_fdot, 
            wdm_lookup_table.min_f_scaled,
            wdm_lookup_table.min_fdot,
            wdm_lookup_table.df,
            wdm_lookup_table.dt,
            wdm_lookup_table.NF,
            wdm_lookup_table.NT,
            wdm_lookup_table.num_channel,
            differential_component,
        )

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None):
        params_tmp = self.xp.atleast_2d(self.xp.asarray(params))
        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        d_h_out = self.xp.zeros(num_bin)
        h_h_out = self.xp.zeros(num_bin)

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
            d_h_out, 
            h_h_out, 
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
        breakpoint()


class GBSTFTComputations(FastLISAResponseParallelModule):
    """STFT-basis analogue of GBWDMComputations.

    Parameters
    ----------
    stft_lookup_table : object
        An ``STFTLookupTable``-like object with attributes:
        ``window_dft`` (complex 1-D array), ``num_delta_f``, ``d_delta_f``,
        ``min_delta_f``, ``window_half_width``, ``df``, ``dt``, ``NF`` (num_m),
        ``NT`` (num_n), ``num_channel``.
    T : float
        Observation time.
    orbits : Orbits or None
        LISA orbit model (defaults to ``EqualArmlengthOrbits``).
    tdi_config : TDIConfig, str, or None
        TDI generation string or instance (defaults to ``"1st generation"``).
    force_backend : str or None
        Force a specific backend (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``).
    """
    def __init__(self, stft_lookup_table, T, orbits=None, tdi_config=None, force_backend=None):
        super().__init__(force_backend=force_backend)
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.stft_lookup_table = stft_lookup_table
        self.T = T

    # ---- properties reused from GBWDMComputations (same pattern) ----

    @property
    def tdi_config(self) -> TDIConfig:
        return self._tdi_config

    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig):
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation")
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config)
        elif not isinstance(tdi_config, TDIConfig):
            raise ValueError("TDI Config needs to be a string or an instance of TDIConfig.")
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
        if orbits is None:
            orbits = EqualArmlengthOrbits()
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            orbits = orbits()
        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)
        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)
        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)

    @property
    def stft_lookup_table(self) -> object:
        return self._stft_lookup_table

    @stft_lookup_table.setter
    def stft_lookup_table(self, stft_lookup_table) -> None:
        """Set STFT lookup table.

        ``stft_lookup_table`` must provide:
        - ``window_dft``      : complex 1-D array (the precomputed window DFT)
        - ``num_delta_f``     : int
        - ``d_delta_f``       : float
        - ``min_delta_f``     : float
        - ``window_half_width``: int
        - ``df``, ``dt``      : float  (STFT bin spacing and segment duration)
        - ``NF`` (num_m), ``NT`` (num_n) : int
        - ``num_channel``     : int
        """
        self._stft_lookup_table = stft_lookup_table
        self.window_dft = stft_lookup_table.window_dft.copy()
        differential_component = stft_lookup_table.differential_component if hasattr(stft_lookup_table, "differential_component") else 1.0
        self.cpp_stft_lookup_table = self.backend.STFTLookupTableWrap(
            self.window_dft,
            stft_lookup_table.num_delta_f,
            stft_lookup_table.d_delta_f,
            stft_lookup_table.min_delta_f,
            stft_lookup_table.window_half_width,
            stft_lookup_table.df,
            stft_lookup_table.dt,
            stft_lookup_table.NF,
            stft_lookup_table.NT,
            stft_lookup_table.num_channel,
            differential_component,
        )

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_ll_stft(self, params, stft_holder, data_index=None, noise_index=None):
        """Compute the log-likelihood in the STFT basis.

        Parameters
        ----------
        params : array-like
            Source parameters, shape ``(num_bin, nparams)`` or ``(nparams,)``.
        stft_holder : object
            Container with ``linear_data_arr`` (list of complex arrays) and
            ``linear_psd_arr`` (list of complex arrays — inverse noise
            covariance, Hermitian), analogous to ``wdm_holder``.
        data_index, noise_index : int array or None
            Per-binary indices into multiple data/noise realisations.

        Returns
        -------
        d_h_out, h_h_out : array
            Inner-product arrays of length ``num_bin``.
        """
        params_tmp = self.xp.atleast_2d(self.xp.asarray(params))
        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        d_h_out = self.xp.zeros(num_bin)
        h_h_out = self.xp.zeros(num_bin)

        num_data = num_noise = len(stft_holder)
        self.cpp_stft = self.backend.STFTDomainWrap(
            stft_holder.linear_data_arr[0],
            stft_holder.linear_psd_arr[0],
            self.stft_lookup_table.df,
            self.stft_lookup_table.dt,
            self.stft_lookup_table.NF,
            self.stft_lookup_table.NT,
            self.tdi_config.nchannels,
            num_data,
            num_noise,
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

        self.backend.GBComputationGroupWrap().gb_stft_get_ll(
            d_h_out,
            h_h_out,
            self.cpp_orbits,
            self.cpp_tdi_config,
            self.cpp_stft_lookup_table,
            self.cpp_stft,
            params_in,
            data_index,
            noise_index,
            num_bin,
            nparams,
            self.T,
            self.backend.TDITypeDict["XYZ"],
        )