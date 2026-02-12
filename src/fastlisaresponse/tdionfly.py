"""TDI-on-the-fly computation module for LISA gravitational-wave response.

This module provides classes for computing Time-Delay Interferometry (TDI)
observables on the fly, supporting both time-domain and frequency-domain
waveform models. It includes GPU-accelerated implementations via the
``gpubackendtools`` infrastructure.

Classes
-------
TDIonTheFly
    Base class for on-the-fly TDI computation.
TDTDIonTheFly
    Time-domain spline-based TDI computation.
FDTDIonTheFly
    Frequency-domain spline-based TDI computation.
GBTDIonTheFly
    Galactic-binary analytic TDI computation.
TDIOutput
    Container for TDI output data.
TDTDIOutput
    Time-domain specialised TDI output container.
FDTDIOutput
    Frequency-domain specialised TDI output container.
"""

from __future__ import annotations

import time
from copy import deepcopy
from math import factorial
from typing import List, Optional, Tuple

import h5py
import numpy as np
from gpubackendtools import wrapper
from gpubackendtools.interpolate import CubicSplineInterpolant
from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.utility import AET
from scipy.interpolate import CubicSpline as CubicSpline_scipy

from .tdiconfig import TDIConfig
from .utils.parallelbase import FastLISAResponseParallelModule

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

# TODO: need to update constants setup
#YRSID_SI = 31558149.763545603
from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI
"""float: Number of SI seconds in a sidereal year."""


def get_factorial(n: int) -> int:
    """Compute the factorial of *n* iteratively.

    Parameters
    ----------
    n : int
        Non-negative integer whose factorial is computed.

    Returns
    -------
    int
        The factorial *n!*.
    """
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


factorials = np.array([factorial(i) for i in range(30)])
"""numpy.ndarray: Pre-computed factorials from 0! to 29!."""

C_inv = 3.3356409519815204e-09
"""float: Inverse of the speed of light in vacuum (1 / c) in SI units [s/m]."""


class CubicSpline:
    """Placeholder alias for the cubic-spline Cython/C++ extension class.

    This class exists so that type checks (e.g. ``isinstance(obj, CubicSpline)``)
    work when the underlying compiled spline class is not directly importable
    from Python.
    """
    pass

class TDIonTheFly(FastLISAResponseParallelModule):
    """Base class for computing LISA TDI observables on the fly.

    Supports GPU acceleration, which is particularly beneficial for
    Bayesian inference workflows.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency of the output time series [Hz].
    num_sub : int
        Number of sub-sources to process in a single batched call.
    n_params : int, optional
        Number of waveform parameters per source.  Default is 4.
    tdi_config : str, list, or TDIConfig, optional
        TDI configuration.  Stock presets are ``'1st generation'``
        and ``'2nd generation'``.  A list of dictionaries describing
        individual TDI link contributions is also accepted (see
        :class:`TDIConfig` for the dictionary format).
        Default is ``'1st generation'``.
    orbits : Orbits or None, optional
        LISA orbital model from LISA Analysis Tools.  Compatible with
        `LISA Orbits <https://lisa-simulation.pages.in2p3.fr/orbits/>`_
        outputs.  Default is :class:`EqualArmlengthOrbits`.
    tdi_chan : str, optional
        TDI channel combination to return: ``'XYZ'``, ``'AET'``, or
        ``'AE'``.  Default is ``'XYZ'``.
    force_backend : str or None, optional
        Force a specific compute backend (``"cpu"``, ``"cuda11x"``,
        ``"cuda12x"``).  Default is ``None`` (auto-detect).
    """

    def __init__(
        self,
        sampling_frequency: float,
        num_sub: int,
        n_params: int = 4,
        tdi_config: Optional[TDIConfig] = None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_chan: str = "XYZ",
        force_backend: Optional[str] = None,
    ):
        """Initialise the TDI-on-the-fly engine.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency of the output time series in Hz.
        num_sub : int
            Number of sub-sources (binary signals) to process in
            a single batched call.
        n_params : int, optional
            Number of waveform parameters per source. Default is 4
            (inclination, polarisation, ecliptic longitude, ecliptic
            latitude).
        tdi_config : TDIConfig or None, optional
            TDI configuration object.  Accepts a :class:`TDIConfig`
            instance or ``None`` (defaults to ``"1st generation"``).
        orbits : Orbits or None, optional
            LISA orbital configuration.  Defaults to
            :class:`EqualArmlengthOrbits`.
        tdi_chan : str, optional
            TDI channel combination to return (``'XYZ'``, ``'AET'``,
            or ``'AE'``).  Default is ``'XYZ'``.
        force_backend : str or None, optional
            Force a specific compute backend
            (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``).
        """
        # Sampling & source configuration
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency
        self.n_params = n_params
        self.num_sub = num_sub

        # TDI channel selection
        self.tdi_chan = tdi_chan
        super().__init__(force_backend=force_backend)

        # Orbit model and TDI configuration
        self.orbits = orbits
        self.tdi_config = tdi_config
        
    @property
    def tdi_config(self) -> TDIConfig:
        """TDIConfig: Current TDI configuration."""
        return self._tdi_config
    
    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig) -> None:
        """Set the TDI configuration.

        Accepts ``None`` (defaults to 1st-generation TDI), a preset
        string (``"1st generation"`` / ``"2nd generation"``), or an
        already-instantiated :class:`TDIConfig`.

        Raises
        ------
        ValueError
            If *tdi_config* is not a valid type.
        """
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
        """Module: Array backend module (NumPy or CuPy)."""
        return self.backend.xp
    
    @property
    def orbits(self) -> Orbits:
        """Orbits: The LISA orbital model used for light-travel-time delays."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        """Set the LISA orbital model.

        Parameters
        ----------
        orbits : Orbits, type, or None
            An :class:`Orbits` instance, an uninitialised :class:`Orbits`
            subclass (instantiated with default arguments), or ``None``
            (defaults to :class:`EqualArmlengthOrbits`).

        Notes
        -----
        The orbits are deep-copied to avoid unintended side-effects.
        If not already configured, :meth:`Orbits.configure` is called
        with ``linear_interp_setup=True``.
        """
        if orbits is None:
            orbits = EqualArmlengthOrbits()
        
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            # Uninitialised subclass – instantiate with default arguments
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)

        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)

        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)
    
    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls) -> list:
        """Return the list of backend identifiers supported by this class.

        Returns
        -------
        list of str
            Supported backend names prefixed with ``"fastlisaresponse_"``.
        """
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, inc, psi, lam, beta, return_spline: bool = False) -> TDIOutput:
        
        params = self.xp.asarray([inc, psi, lam, beta]).T.flatten().copy()

        assert len(params) == 4 * self.num_sub

        tdi_channels_arr = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=complex)
        tdi_amp = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        tdi_phase = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        phase_ref = self.xp.zeros((self.N * self.num_sub), dtype=float)
        assert int(np.prod(self.t_arr.shape)) == self.N * self.num_sub

        self.wave_gen.run_wave_tdi_wrap(
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phase_ref,
            params, self.t_arr.flatten().copy(),
            self.N, self.num_sub, self.n_params, self.tdi_config.nchannels
        )
        
        breakpoint()
        reshape_shape = (self.num_sub, self.tdi_config.nchannels, self.N)
        return self.from_tdi_output(TDIOutput(
            self.t_arr, 
            tdi_amp.reshape(reshape_shape), 
            tdi_phase.reshape(reshape_shape), 
            phase_ref.reshape(self.t_arr.shape)
        ), fill_splines=return_spline)
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        return tdi_output


CUBIC_SPLINE_LINEAR_SPACING = 1
"""int: Constant indicating linearly-spaced cubic-spline knots."""

CUBIC_SPLINE_LOG10_SPACING = 2
"""int: Constant indicating log10-spaced cubic-spline knots."""

CUBIC_SPLINE_GENERAL_SPACING = 3
"""int: Constant indicating arbitrarily-spaced cubic-spline knots."""


class TDTDIonTheFly(TDIonTheFly):
    """Time-domain TDI-on-the-fly using cubic-spline waveform representations.

    This class wraps a time-domain amplitude and phase (provided as arrays
    or cubic splines) to compute TDI observables on the fly.

    Parameters
    ----------
    t : numpy.ndarray
        Evaluation times at which TDI observables will be computed [s].
    amp : numpy.ndarray, CubicSpline_scipy, or CubicSpline
        Waveform amplitude.  If an array, a :class:`CubicSplineInterpolant`
        is built internally using *t_input* as the knot times.
    phase : numpy.ndarray, CubicSpline_scipy, or CubicSpline
        Waveform phase [rad].  Must be the same type as *amp*.
    *args
        Positional arguments forwarded to :class:`TDIonTheFly`.
    t_input : numpy.ndarray or None, optional
        Knot times for building splines when *amp*/*phase* are arrays.
        Required when raw arrays are passed.
    **kwargs
        Keyword arguments forwarded to :class:`TDIonTheFly`.
    """

    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSpline,
        phase: np.ndarray | CubicSpline_scipy | CubicSpline,
        *args, 
        t_input: Optional[np.ndarray] = None, 
        **kwargs
    ): 
        super().__init__(*args, **kwargs)

        self.phase_input = phase
        self.amp_input = amp

        if isinstance(amp, np.ndarray) or isinstance(amp, cp.ndarray):
            if isinstance(amp, np.ndarray):
                assert isinstance(phase, np.ndarray) and isinstance(t, np.ndarray)
                assert t_input is not None and isinstance(t_input, np.ndarray)
            else:
                assert isinstance(phase, cp.ndarray) and isinstance(t, cp.ndarray)
                assert t_input is not None and isinstance(t_input, cp.ndarray)
            
            self.spline_length = len(phase)

            t_input = self.xp.atleast_2d(self.xp.asarray(t_input))
            
            if t_input.shape[0] == 1:
                t_input = self.xp.repeat(t_input, amp.shape[0], axis=0)

            amp = self.xp.atleast_2d(self.xp.asarray(amp))
            phase = self.xp.atleast_2d(self.xp.asarray(phase))

            # TODO: improve when gbt is fixed up
            amp = CubicSplineInterpolant(t_input.copy(), amp, force_backend=self.backend.name.split("_")[-1])
            phase = CubicSplineInterpolant(t_input.copy(), phase, force_backend=self.backend.name.split("_")[-1])

        elif isinstance(amp, CubicSpline_scipy):
            raise NotImplementedError
            assert isinstance(phase, CubicSpline_scipy)

            self.spline_length = phase.c.shape[-1] + 1

            phase_y = phase.c[3, :].copy()
            phase_c1 = phase.c[2, :].copy()
            phase_c2 = phase.c[1, :].copy()
            phase_c3 = phase.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = amp.c[2, :].copy()
            amp_c2 = amp.c[1, :].copy()
            amp_c3 = amp.c[0, :].copy()

            breakpoint()
            # x = amp

            # convert to pointers
            targs, twkargs = wrapper(t, phase_y, phase_c1, phase_c2, phase_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            phase = self.backend.pyCubicSplineWrap(_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSplineInterpolant):
            assert isinstance(phase, CubicSplineInterpolant)

        else:
            raise ValueError("# TODO: fix this.")
        
        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))

        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
        self.amp = amp
        self.phase = phase

        # self.wave_gen = self.backend.pyTDSplineTDIWaveform()
        # self.wave_gen.add_orbit_information(*self.orbits.pycppdetector_args)
        # self.wave_gen.add_tdi_config(*self.tdi_config.pytdiconfig_args)
        # self.wave_gen.add_amp_spline(*self.amp.cpp_class_args)
        # self.wave_gen.add_phase_spline(*self.phase.cpp_class_args)
        
        # import time
        # time.sleep(1.0)
    @property
    def wave_gen(self) -> callable:
        """callable: The C++/CUDA waveform generator wrapper."""
        self.cpp_amp = self.backend.CubicSplineWrap(*self.amp.cpp_class_args)
        self.cpp_phase = self.backend.CubicSplineWrap(*self.phase.cpp_class_args)
        self._wave_gen = self.backend.TDSplineTDIWaveformWrap(self.cpp_orbits, self.cpp_tdi_config, self.cpp_amp, self.cpp_phase)
        return self._wave_gen
    
    # @wave_gen.setter
    # def wave_gen(self, wave_gen) -> None:
    #     self._wave_gen = wave_gen
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> "TDTDIOutput":
        """Wrap a base :class:`TDIOutput` as a :class:`TDTDIOutput`.

        Parameters
        ----------
        tdi_output : TDIOutput
            Raw TDI output to convert.
        fill_splines : bool, optional
            Whether to build spline interpolants.  Default is ``False``.

        Returns
        -------
        TDTDIOutput
        """
        assert self.xp.allclose(tdi_output.x, self.t_arr)
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )
    

class TDIOutput(FastLISAResponseParallelModule):
    """Container for TDI computation output.

    Stores the evaluation points, TDI amplitudes, TDI phases, and
    reference phases for each channel and source.  Optionally builds
    cubic-spline interpolants that allow efficient evaluation at
    arbitrary points.

    Parameters
    ----------
    x : numpy.ndarray
        Independent-variable array (time or frequency) with shape
        ``(num_bin, N)``.
    tdi_amp : numpy.ndarray
        TDI amplitude array with shape ``(num_bin, nchannels, N)``.
    tdi_phase : numpy.ndarray
        TDI phase array with shape ``(num_bin, nchannels, N)`` [rad].
    phase_ref : numpy.ndarray
        Reference phase array with shape ``(num_bin, N)`` [rad].
    fill_splines : bool, optional
        If ``True``, cubic-spline interpolants are built for
        *tdi_amp*, *tdi_phase*, and *phase_ref* upon assignment.
        Default is ``True``.
    **kwargs
        Extra keyword arguments forwarded to
        :class:`FastLISAResponseParallelModule`.
    """

    def __init__(self, x, tdi_amp, tdi_phase, phase_ref, fill_splines=True, **kwargs):
        
        self.fill_splines = fill_splines
        if self.fill_splines:
            self._splines = {}

        self.x = x
        super().__init__(**kwargs)

        # need to be after for proper setter
        self.tdi_amp, self.tdi_phase = tdi_amp, tdi_phase
        self.phase_ref = phase_ref
       
        
    def _get_spl(self, key: str) -> CubicSplineInterpolant:
        """Retrieve a pre-built spline by name.

        Parameters
        ----------
        key : str
            Key into the internal spline dictionary (e.g.
            ``"tdi_amp"``, ``"tdi_phase"``, ``"phase_ref"``).

        Returns
        -------
        CubicSplineInterpolant

        Raises
        ------
        AssertionError
            If ``fill_splines`` was ``False`` at construction time.
        """
        assert self.fill_splines
        return self._splines[key]
    
    @property
    def phase_ref_spl(self) -> CubicSplineInterpolant:
        """CubicSplineInterpolant: Spline interpolant for the reference phase."""
        return self._get_spl("phase_ref")
    
    def build_spline(self, x: np.ndarray, y: np.ndarray, **kwargs) -> CubicSplineInterpolant:
        """Build a :class:`CubicSplineInterpolant` from *x* and *y*.

        If *x* is 2-D and *y* is 3-D (i.e. multi-channel), *x* is
        broadcast along the channel axis before constructing the spline.

        Parameters
        ----------
        x : numpy.ndarray
            Independent-variable array.
        y : numpy.ndarray
            Dependent-variable array.
        **kwargs
            Extra arguments forwarded to :class:`CubicSplineInterpolant`.

        Returns
        -------
        CubicSplineInterpolant
        """
        if x.ndim == 2 and y.ndim == 3:
            x_in = self.xp.repeat(x[:, None, :], y.shape[1], axis=1)
        else:
            x_in = x.copy()
        return CubicSplineInterpolant(x_in, y, **kwargs, force_backend=self.backend.name.split("_")[-1])
    
    @property
    def num_bin(self) -> int:
        """int: Number of independent source bins in the output."""
        if self.tdi_amp.ndim == 3:
            return self.tdi_amp.shape[0]
        elif self.tdi_amp.ndim == 2:
            return 1

    @classmethod
    def supported_backends(cls) -> list:
        """Return the list of supported backend identifiers."""
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def X(self) -> np.ndarray:
        """numpy.ndarray: Complex X-channel TDI observable."""
        return self.Xamp * self.xp.exp(-1j * (self.Xphase + self.phase_ref))

    @property
    def Y(self) -> np.ndarray:
        """numpy.ndarray: Complex Y-channel TDI observable."""
        return self.Yamp * self.xp.exp(-1j * (self.Yphase + self.phase_ref))

    @property
    def Z(self) -> np.ndarray:
        """numpy.ndarray: Complex Z-channel TDI observable."""
        return self.Zamp * self.xp.exp(-1j * (self.Zphase + self.phase_ref))

    @property
    def Xamp(self) -> np.ndarray:
        """numpy.ndarray: Amplitude envelope for the X channel."""
        return self.tdi_amp[:, 0]

    @property
    def Yamp(self) -> np.ndarray:
        """numpy.ndarray: Amplitude envelope for the Y channel."""
        return self.tdi_amp[:, 1]

    @property
    def Zamp(self) -> np.ndarray:
        """numpy.ndarray: Amplitude envelope for the Z channel."""
        return self.tdi_amp[:, 2]

    @property
    def Xphase(self) -> np.ndarray:
        """numpy.ndarray: Phase for the X channel [rad]."""
        return self.tdi_phase[:, 0]

    @property
    def Yphase(self) -> np.ndarray:
        """numpy.ndarray: Phase for the Y channel [rad]."""
        return self.tdi_phase[:, 1]

    @property
    def Zphase(self) -> np.ndarray:
        """numpy.ndarray: Phase for the Z channel [rad]."""
        return self.tdi_phase[:, 2]
    
    @property
    def tdi_amp(self) -> np.ndarray:
        """numpy.ndarray: TDI amplitude array."""
        return self._tdi_amp
    
    @tdi_amp.setter
    def tdi_amp(self, tdi_amp: np.ndarray) -> None:
        if self.fill_splines:
            self._splines["tdi_amp"] = self.build_spline(self.x, tdi_amp)
        self._tdi_amp = tdi_amp

    @property
    def tdi_phase(self) -> np.ndarray:
        """numpy.ndarray: TDI phase array [rad]."""
        return self._tdi_phase
    
    @tdi_phase.setter
    def tdi_phase(self, tdi_phase: np.ndarray) -> None:
        if self.fill_splines:
            self._splines["tdi_phase"] = self.build_spline(self.x, tdi_phase)
        self._tdi_phase = tdi_phase

    @property
    def phase_ref(self) -> np.ndarray:
        """numpy.ndarray: Reference phase array [rad]."""
        return self._phase_ref
    
    @phase_ref.setter
    def phase_ref(self, phase_ref: np.ndarray) -> None:
        if self.fill_splines:
            self._splines["phase_ref"] = self.build_spline(self.x, phase_ref)
        self._phase_ref = phase_ref
    
    @property
    def tdi_amp_spl(self) -> CubicSplineInterpolant:
        """CubicSplineInterpolant: Spline interpolant for the TDI amplitude."""
        return self._get_spl("tdi_amp")
    
    @property
    def tdi_phase_spl(self) -> CubicSplineInterpolant:
        """CubicSplineInterpolant: Spline interpolant for the TDI phase."""
        return self._get_spl("tdi_phase")
    
    @property
    def Aamp(self) -> np.ndarray:
        """numpy.ndarray: Amplitude of the A-channel (AET basis). Not yet implemented."""
        raise NotImplementedError

    @property
    def Aphase(self) -> np.ndarray:
        """numpy.ndarray: Phase of the A-channel (AET basis). Not yet implemented."""
        raise NotImplementedError

    @property
    def Eamp(self) -> np.ndarray:
        """numpy.ndarray: Amplitude of the E-channel (AET basis). Not yet implemented."""
        raise NotImplementedError

    @property
    def Ephase(self) -> np.ndarray:
        """numpy.ndarray: Phase of the E-channel (AET basis). Not yet implemented."""
        raise NotImplementedError

    @property
    def Tamp(self) -> np.ndarray:
        """numpy.ndarray: Amplitude of the T-channel (AET basis). Not yet implemented."""
        raise NotImplementedError

    @property
    def Tphase(self) -> np.ndarray:
        """numpy.ndarray: Phase of the T-channel (AET basis). Not yet implemented."""
        raise NotImplementedError
    
    def eval_spline_vals(self, x_new: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the stored splines at new independent-variable values.

        Parameters
        ----------
        x_new : numpy.ndarray
            New evaluation points.  1-D arrays are broadcast across all
            source bins; 2-D arrays are used directly (one row per bin).
        **kwargs
            Extra arguments forwarded to the spline ``__call__`` method.

        Returns
        -------
        tuple of numpy.ndarray
            ``(tdi_amp_new, tdi_phase_new, phase_ref_new)`` evaluated at
            *x_new*.
        """
        if x_new.ndim == 1:
            t_amp_phase = self.xp.tile(x_new, (self.num_bin, 3, 1))
            t_phase_ref = self.xp.tile(x_new, (self.num_bin, 1))
        elif x_new.ndim == 2:
            t_amp_phase = self.xp.repeat(x_new[:, None, :], 3, axis=1)
            t_phase_ref = x_new

        tdi_amp_new = self.tdi_amp_spl(t_amp_phase, **kwargs)
        tdi_phase_new = self.tdi_phase_spl(t_amp_phase, **kwargs)
        phase_ref_new = self.phase_ref_spl(t_phase_ref, **kwargs)
        return (tdi_amp_new, tdi_phase_new, phase_ref_new)
    
    def eval_tdi(self, x_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the reconstructed TDI observable at new points.

        Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
    
class TDTDIOutput(TDIOutput):
    """Time-domain specialised TDI output container.

    Provides :meth:`eval_tdi` for reconstructing real-valued TDI time
    series at arbitrary time samples via spline interpolation.
    """

    @classmethod
    def from_tdi_output(cls, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> "TDTDIOutput":
        """Construct a :class:`TDTDIOutput` from an existing :class:`TDIOutput`.

        Parameters
        ----------
        tdi_output : TDIOutput
            Source output container.
        fill_splines : bool, optional
            Whether to build spline interpolants.  Default is ``False``.

        Returns
        -------
        TDTDIOutput
        """
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )

    def eval_tdi(self, t_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the real-valued TDI time series at new time samples.

        Parameters
        ----------
        t_new : numpy.ndarray
            New evaluation times [s].
        **kwargs
            Forwarded to :meth:`eval_spline_vals`.

        Returns
        -------
        numpy.ndarray
            Real-valued TDI observable(s) at *t_new*.
        """
        tdi_amp_new, tdi_phase_new, phase_ref_new = self.eval_spline_vals(t_new, **kwargs)
        tdi_output = self.xp.real(tdi_amp_new * self.xp.exp(-1j * (tdi_phase_new + phase_ref_new[:, None, :])))
        return tdi_output
    
    @property
    def t_arr(self) -> np.ndarray:
        """numpy.ndarray: Alias for :attr:`x` (time samples)."""
        return self.x
        
    
class FDTDIOutput(TDIOutput):
    """Frequency-domain specialised TDI output container.

    Provides :meth:`eval_tdi` for reconstructing complex-valued TDI
    frequency series at arbitrary frequency samples.
    """

    def eval_tdi(self, f_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the complex TDI frequency series at new frequencies.

        Parameters
        ----------
        f_new : numpy.ndarray
            New evaluation frequencies [Hz].
        **kwargs
            Forwarded to :meth:`eval_spline_vals`.

        Returns
        -------
        numpy.ndarray
            Complex TDI observable(s) at *f_new*.
        """
        tdi_amp_new, tdi_phase_new, phase_ref_new = self.eval_spline_vals(f_new, **kwargs)
        tdi_output = tdi_amp_new * self.xp.exp(-1j * (tdi_phase_new + phase_ref_new[:, None, :]))
        return tdi_output
    
    @property
    def f_arr(self) -> np.ndarray:
        """numpy.ndarray: Alias for :attr:`x` (frequency samples)."""
        return self.x


# TODO: make it log spaced in frequency?

class FDTDIonTheFly(TDIonTheFly):
    """Frequency-domain TDI-on-the-fly using cubic-spline waveform representations.

    The waveform is parameterised in terms of an amplitude spline and a
    frequency spline (both functions of time), plus a reference phase.
    TDI observables are computed internally via the C++/CUDA
    ``FDSplineTDIWaveform`` engine.

    Parameters
    ----------
    t : numpy.ndarray
        Evaluation times at which TDI observables are computed [s].
    amp : numpy.ndarray, CubicSpline_scipy, or CubicSplineInterpolant
        Waveform amplitude as a function of time.
    freq : numpy.ndarray, CubicSpline_scipy, or CubicSplineInterpolant
        Instantaneous gravitational-wave frequency as a function of
        time [Hz].
    phase_ref : numpy.ndarray, CubicSpline_scipy, or CubicSplineInterpolant
        Reference phase as a function of time [rad].
    *args
        Positional arguments forwarded to :class:`TDIonTheFly`.
    t_input : numpy.ndarray or None, optional
        Spline knot times when raw arrays are passed.  Required if
        *amp*/*freq* are plain arrays.
    spline_type : int, optional
        Knot-spacing type for the splines (see module-level constants).
        Default is ``CUBIC_SPLINE_GENERAL_SPACING``.
    force_backend : str or None, optional
        Force a specific compute backend.
    **kwargs
        Keyword arguments forwarded to :class:`TDIonTheFly`.
    """

    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        freq: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        phase_ref: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        *args, 
        t_input: Optional[np.ndarray] = None, 
        spline_type: int = CUBIC_SPLINE_GENERAL_SPACING,
        force_backend: str = None,
        **kwargs
    ): 
        super().__init__(*args, force_backend=force_backend, **kwargs)

        self.freq_input = freq
        self.amp_input = amp
        self.phase_ref = phase_ref
        
        if isinstance(amp, np.ndarray) or isinstance(amp, cp.ndarray):
            if isinstance(amp, np.ndarray):
                assert isinstance(freq, np.ndarray) and isinstance(t, np.ndarray)
                assert t_input is not None and isinstance(t_input, np.ndarray)
            else:
                assert isinstance(freq, cp.ndarray) and isinstance(t, cp.ndarray)
                assert t_input is not None and isinstance(t_input, cp.ndarray)
            
            self.spline_length = len(freq)

            t_input = self.xp.atleast_2d(self.xp.asarray(t_input))
            
            if t_input.shape[0] == 1:
                t_input = self.xp.repeat(t_input, amp.shape[0], axis=0)

            amp = self.xp.atleast_2d(self.xp.asarray(amp))
            freq = self.xp.atleast_2d(self.xp.asarray(freq))

            # TODO: improve when gbt is fixed up
            amp = CubicSplineInterpolant(t_input.copy(), amp, force_backend=self.backend.name.split("_")[-1])
            freq = CubicSplineInterpolant(t_input.copy(), freq, force_backend=self.backend.name.split("_")[-1])
            

        elif isinstance(amp, CubicSpline_scipy):
            assert isinstance(freq, CubicSpline_scipy)
            raise NotImplementedError
            self.spline_length = freq.c.shape[-1] + 1

            freq_y = freq.c[3, :].copy()
            freq_c1 = freq.c[2, :].copy()
            freq_c2 = freq.c[1, :].copy()
            freq_c3 = freq.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = amp.c[2, :].copy()
            amp_c2 = amp.c[1, :].copy()
            amp_c3 = amp.c[0, :].copy()

            breakpoint()
            # x = amp

            # convert to pointers
            targs, twkargs = wrapper(t, freq_y, freq_c1, freq_c2, freq_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            freq = self.backend.pyCubicSplineWrap(_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSplineInterpolant):
            assert isinstance(freq, CubicSplineInterpolant)
            # f = freq.y, t = freq.x

        else:
            raise ValueError("# TODO: fix this.")
        
        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))
        
        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
        self.amp = amp
        self.freq = freq

    @property
    def wave_gen(self) -> callable:
        """callable: Lazily-constructed C++/CUDA waveform generator.

        Each access rebuilds the wrapper objects from the current
        amplitude and frequency splines so that parameter updates are
        always reflected.
        """
        self.cpp_amp = self.backend.CubicSplineWrap(*self.amp.cpp_class_args)
        self.cpp_freq = self.backend.CubicSplineWrap(*self.freq.cpp_class_args)
        self._wave_gen = self.backend.FDSplineTDIWaveformWrap(self.cpp_orbits, self.cpp_tdi_config, self.cpp_amp, self.cpp_freq)
        return self._wave_gen
    
    @property
    def spline_type(self) -> int:
        """int: Knot-spacing type for the internal splines."""
        return self._spline_type
    
    @spline_type.setter
    def spline_type(self, spline_type: int) -> None:
        assert isinstance(spline_type, int)
        assert spline_type in [CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING]
        self._spline_type = spline_type
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        """Wrap a base :class:`TDIOutput` as a :class:`FDTDIOutput`.

        The independent variable is converted from time to frequency
        using the stored frequency spline.

        Parameters
        ----------
        tdi_output : TDIOutput
            Raw TDI output.
        fill_splines : bool, optional
            Whether to build spline interpolants.  Default is ``False``.

        Returns
        -------
        FDTDIOutput
        """
        # TODO: remove the freq spline?
        return FDTDIOutput(
            self.freq(tdi_output.x), tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )
    

class GBTDIonTheFly(TDIonTheFly):
    """Galactic-binary (GB) TDI-on-the-fly computation.

    Uses the analytic galactic-binary waveform model (amplitude, frequency,
    and frequency derivative are simple functions of the UCB parameters)
    to compute TDI observables directly without pre-computed splines.

    Parameters
    ----------
    t : numpy.ndarray
        Evaluation times [s].
    T : float
        Total observation duration [s].
    *args
        Positional arguments forwarded to :class:`TDIonTheFly`.
    **kwargs
        Keyword arguments forwarded to :class:`TDIonTheFly`.
    """

    def __init__(self, 
        t: np.ndarray,
        T: float,
        *args, 
        **kwargs
    ): 
        super().__init__(*args, **kwargs)

        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))
        self.T = T
        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
    @property
    def wave_gen(self) -> callable:
        """callable: Lazily-constructed C++/CUDA GB waveform generator."""
        return self._wave_gen
    
    @wave_gen.setter
    def wave_gen(self, wave_gen) -> None:
        self._wave_gen = wave_gen
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> "TDTDIOutput":
        """Wrap a base :class:`TDIOutput` as a :class:`TDTDIOutput`.

        Parameters
        ----------
        tdi_output : TDIOutput
            Raw TDI output.
        fill_splines : bool, optional
            Whether to build spline interpolants.  Default is ``False``.

        Returns
        -------
        TDTDIOutput
        """
        assert self.xp.allclose(tdi_output.x, self.t_arr)
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )
    
    @property
    def wave_gen(self) -> callable:
        """callable: Rebuild the GB TDI wrapper each time it is accessed."""
        self._wave_gen = self.backend.GBTDIonTheFlyWrap(self.cpp_orbits, self.cpp_tdi_config, self.T)
        return self._wave_gen
    
    def __call__(self, amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta, return_spline: bool = False) -> TDIOutput:
        """Compute TDI observables for one or more galactic binaries.

        Parameters
        ----------
        amp : array_like
            Gravitational-wave amplitude(s).
        f0 : array_like
            Initial frequency(ies) [Hz].
        fdot0 : array_like
            First frequency derivative(s) [Hz/s].
        fddot0 : array_like
            Second frequency derivative(s) [Hz/s²].
        phi0 : array_like
            Initial phase(s) [rad].
        inc : array_like
            Inclination(s) [rad].
        psi : array_like
            Polarisation angle(s) [rad].
        lam : array_like
            Ecliptic longitude(s) [rad].
        beta : array_like
            Ecliptic latitude(s) [rad].
        return_spline : bool, optional
            If ``True``, build spline interpolants on the output.
            Default is ``False``.

        Returns
        -------
        TDIOutput
            Container with computed TDI amplitudes, phases, and
            reference phases.
        """
        params = self.xp.asarray([amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta]).T.flatten().copy()

        assert len(params) == 9 * self.num_sub

        tdi_channels_arr = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=complex)
        tdi_amp = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        tdi_phase = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        phase_ref = self.xp.zeros((self.N * self.num_sub), dtype=float)
        assert int(np.prod(self.t_arr.shape)) == self.N * self.num_sub

        self.wave_gen.run_wave_tdi_wrap(
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phase_ref,
            params, self.t_arr.flatten().copy(),
            self.N, self.num_sub, self.n_params, self.tdi_config.nchannels
        )

        reshape_shape = (self.num_sub, self.tdi_config.nchannels, self.N)
        return self.from_tdi_output(TDIOutput(
            self.t_arr, 
            tdi_amp.reshape(reshape_shape), 
            tdi_phase.reshape(reshape_shape), 
            phase_ref.reshape(self.t_arr.shape)
        ), fill_splines=return_spline)