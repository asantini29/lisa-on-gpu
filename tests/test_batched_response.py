import sys
import os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

import unittest
import numpy as np
import warnings

from lisatools.detector import EqualArmlengthOrbits
from fastlisaresponse import ResponseWrapper, pyResponseTDI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.utils.parallelbase import FastLISAResponseParallelModule

YRSID_SI = 31558149.763545603


class GBWave(FastLISAResponseParallelModule):
    @property
    def xp(self):
        return self.backend.xp

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)
        fddot = 11.0 / 3.0 * fdot**2 / f
        phase = 2 * np.pi * (f * t + 0.5 * fdot * t**2 + (1.0 / 6.0) * fddot * t**3) - phi0
        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota**2)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi
        return hp + 1j * hc


def _make_wrapper(tdi_gen="1st generation"):
    """Return a configured ResponseWrapper (CPU, remove_sky_coords=True, AET)."""
    T = 2.0
    dt = 10.0
    t_buffer = 10000.0
    order = 25
    orbits = EqualArmlengthOrbits()
    orbits.configure(linear_interp_setup=True)
    gb = GBWave(force_backend="cpu")
    return ResponseWrapper(
        gb, T, dt,
        index_lambda=6, index_beta=7,
        t_buffer=t_buffer,
        flip_hx=False,
        force_backend="cpu",
        remove_sky_coords=True,
        is_ecliptic_latitude=True,
        remove_garbage=True,
        orbits=orbits,
        order=order,
        tdi=TDIConfig(tdi_gen),
        tdi_chan="AET",
    )


# Fixed GB source parameters (sky-independent)
_GB_ARGS = dict(A=1.084702251e-22, f=2.35962078e-3, fdot=1.47197271e-17,
                iota=1.11820901, phi0=4.91128699, psi=2.3290324)

# Three distinct sky positions
_SKY = [
    (5.22979888,  0.9805742971871619),
    (1.5,        -0.3),
    (3.2,         0.6),
    (2.1,         0.1),
]

_LAMBDAS = np.array([pos[0] for pos in _SKY])
_BETAS = np.array([pos[1] for pos in _SKY])

T, dt = 2.0, 10.0

all_polarizations = []

wave_gen = GBWave(force_backend="cpu")

for lam, beta in _SKY:
    polarizations = wave_gen(
        _GB_ARGS["A"], _GB_ARGS["f"], _GB_ARGS["fdot"],
        _GB_ARGS["iota"], _GB_ARGS["phi0"], _GB_ARGS["psi"],
        T=T, dt=dt,
    )
    all_polarizations.append(polarizations)

all_polarizations = np.array(all_polarizations)  # shape (3, num_time_samples)

t = np.arange(0.0, T * YRSID_SI, dt)

t_buffer = 10000.0
order = 25
orbits = EqualArmlengthOrbits()
orbits.configure(linear_interp_setup=True)
tdi_gen = "1st generation"

response = pyResponseTDI(
        num_pts=len(t),
        sampling_frequency=1/dt,
        orbits=orbits,
        order=order,
        tdi=tdi_gen,
        tdi_chan="AET",
)

class TestBatchedResponse(unittest.TestCase):
    def generate_batched(self): 
        
        response.get_projections(all_polarizations, _LAMBDAS, _BETAS, t_buffer=t_buffer)
        
        channels_out = response.get_tdi_delays()

        self.assertEqual(channels_out[0].shape, (len(_SKY), len(t)))

        breakpoint()
if __name__ == "__main__":

    test = TestBatchedResponse()
    test.generate_batched()

    unittest.main()

