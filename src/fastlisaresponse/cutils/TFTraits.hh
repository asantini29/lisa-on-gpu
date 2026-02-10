#ifndef __TF_TRAITS_HH__
#define __TF_TRAITS_HH__

/**
 * @file TFTraits.hh
 * @brief Arithmetic helper overloads for generic time-frequency inner products.
 *
 * These overloads allow the same templated code to compute inner products
 * whether the time-frequency coefficients are real (WDM) or complex (STFT).
 *
 * The noise weighting S^{-1} is included in the inner product definition
 * because in the STFT basis the inverse-noise covariance is complex
 * (Hermitian) and cannot be factored out of the Re(...) operation.
 *
 * Per-pixel inner product contributions:
 *
 *   <d|h>:  d * h * S_inv                        (real / WDM)
 *           Re( conj(d) * S_inv * h )             (complex / STFT)
 *
 *   <h|h>:  h_i * h_j * S_inv                    (real / WDM)
 *           Re( conj(h_i) * S_inv * h_j )         (complex / STFT)
 *
 * By using function overloading (not template specialization), the compiler
 * automatically selects the correct version based on the argument types.
 * CUDA fully supports this in device code.
 */

#include "cuda_complex.hpp"
#include "gbt_global.h"

// =====================================================================
// ip_dh: inner product <d|h> per pixel (noise-weighted)
// =====================================================================

/**
 * @brief Real (WDM) case: d * h * S_inv.
 */
CUDA_CALLABLE_MEMBER inline double ip_dh(double d, double h, double noise, double differential_component) {
    return d * h * noise * differential_component;
}

/**
 * @brief Complex (STFT) case: Re( conj(d) * S_inv * h ).
 *
 * Written out to avoid a separate conj() call:
 *   conj(d) * noise * h  =  (d_r - i d_i)(n_r + i n_i)(h_r + i h_i)
 * We only need the real part of the triple product.
 */
CUDA_CALLABLE_MEMBER inline double ip_dh(cmplx d, cmplx h, cmplx noise, double differential_component) {
    // conj(d) * h
    double re_dh = d.real() * h.real() + d.imag() * h.imag();
    double im_dh = d.real() * h.imag() - d.imag() * h.real();
    // Re( (re_dh + i im_dh) * noise )
    return (re_dh * noise.real() - im_dh * noise.imag()) * differential_component;
}

// =====================================================================
// ip_hh: inner product <h|h> per pixel (noise-weighted)
// =====================================================================

/**
 * @brief Real (WDM) case: h_i * h_j * S_inv.
 */
CUDA_CALLABLE_MEMBER inline double ip_hh(double h_i, double h_j, double noise, double differential_component) {
    return h_i * h_j * noise * differential_component;
}

/**
 * @brief Complex (STFT) case: Re( conj(h_i) * S_inv * h_j ).
 */
CUDA_CALLABLE_MEMBER inline double ip_hh(cmplx h_i, cmplx h_j, cmplx noise, double differential_component) {
    // conj(h_i) * h_j
    double re_hh = h_i.real() * h_j.real() + h_i.imag() * h_j.imag();
    double im_hh = h_i.real() * h_j.imag() - h_i.imag() * h_j.real();
    // Re( (re_hh + i im_hh) * noise )
    return (re_hh * noise.real() - im_hh * noise.imag()) * differential_component;
}

#endif // __TF_TRAITS_HH__
