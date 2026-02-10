#ifndef __TDI_ON_THE_FLY_HH__
#define __TDI_ON_THE_FLY_HH__

#include "Detector.hpp"
#include "Interpolate.hh"
#include "LISAResponse.hh"
#include "gbt_global.h"
#include "TFTraits.hh"


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBTDIonTheFly GBTDIonTheFlyGPU
#define FDSplineTDIWaveform FDSplineTDIWaveformGPU
#define TDSplineTDIWaveform TDSplineTDIWaveformGPU
#define WaveletLookupTable WaveletLookupTableGPU
#define WDMDomain WDMDomainGPU
#define STFTDomain STFTDomainGPU
#define STFTLookupTable STFTLookupTableGPU
#define GBComputationGroup GBComputationGroupGPU
#else
#define GBTDIonTheFly GBTDIonTheFlyCPU
#define FDSplineTDIWaveform FDSplineTDIWaveformCPU
#define TDSplineTDIWaveform TDSplineTDIWaveformCPU
#define WaveletLookupTable WaveletLookupTableCPU
#define WDMDomain WDMDomainCPU
#define STFTDomain STFTDomainCPU
#define STFTLookupTable STFTLookupTableCPU
#define GBComputationGroup GBComputationGroupCPU
#endif

#define TDI_XYZ 1
#define TDI_AET 2
#define TDI_AE 3

class LISATDIonTheFly{
    public:
        Orbits *orbits;
        TDIConfig *tdi_config;
        int inc_index;
        int psi_index;
        int lam_index;
        int beta_index;

        CUDA_DEVICE 
        void run_wave_tdi(
            void *buffer, int buffer_length, cmplx *tdi_channels_arr, 
            double *tdi_amp, double *tdi_phase, double *phi_ref, 
            double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        );
        CUDA_CALLABLE_MEMBER 
        LISATDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, int inc_index_, int psi_index_, int lam_index_, int beta_index_)
        {
            orbits = orbits_;
            tdi_config = tdi_config_;
            inc_index = inc_index_;
            psi_index = psi_index_;
            lam_index = lam_index_;
            beta_index = beta_index_;
        };
        CUDA_CALLABLE_MEMBER 
        ~LISATDIonTheFly();
        CUDA_DEVICE
        void print_orbits_tdi();
        CUDA_DEVICE
        void fill_link_arrays(int *link_space_craft_rec, int *link_space_craft_em);
        // CUDA_DEVICE
        // void LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k);
        CUDA_DEVICE
        void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
        // CUDA_DEVICE
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i);
        // CUDA_DEVICE
        // void get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double t, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        // CUDA_DEVICE
        // void get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i);
        // CUDA_DEVICE
        // void get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n);
        CUDA_DEVICE
        void get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_space_craft_rec, int *link_space_craft_em, Vec k, Vec u, Vec v);
        CUDA_DEVICE
        void get_tdi_Xf_single(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_space_craft_rec, int *link_space_craft_em, int bin_i);
        CUDA_DEVICE
        void extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
        CUDA_DEVICE
        void new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR);
        int get_tdi_buffer_size(int N);
        CUDA_DEVICE
        void unwrap_phase(int N, double *phase);
        CUDA_DEVICE
        void new_unwrap_phase(double *ph_correct_buffer, int N, double *phase);
        CUDA_DEVICE
        void new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr);
        CUDA_DEVICE
        double get_phase_ref(double t, double *params, int bin_i);
        CUDA_DEVICE
        void get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i);
        CUDA_DEVICE
        void get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params);
        CUDA_DEVICE
        void xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n);
        CUDA_DEVICE
        virtual double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_fdot(double t, double *params, int bin_i);
};

class GBTDIonTheFly : public LISATDIonTheFly{
    public:
        double T;
        int amplitude_index;
        int f0_index;
        int fdot0_index;
        int fddot0_index;
        int phi0_index;

        CUDA_CALLABLE_MEMBER
        GBTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_) : LISATDIonTheFly(orbits_, tdi_config_, 5, 6, 7, 8)
        {
            T = T_;
            amplitude_index = 0;
            f0_index = 1;
            fdot0_index = 2;
            fddot0_index = 3;
            phi0_index = 4;
        };
        CUDA_CALLABLE_MEMBER
        ~GBTDIonTheFly();
        // CUDA_DEVICE
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i) override;
        CUDA_DEVICE
        double ucb_amplitude(double t, double *params);
        CUDA_DEVICE
        double ucb_phase(double t, double *params);
        CUDA_DEVICE
        double ucb_fdot(double t, double *params);
        CUDA_DEVICE
        double ucb_f(double t, double *params);
        int get_gb_buffer_size(int N);
        // CUDA_DEVICE
        // void run_wave_tdi(
        //     cmplx *tdi_channels_arr, 
        //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
        //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        // );
        CUDA_DEVICE
        double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_fdot(double t, double *params, int bin_i);
};

void gb_run_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


class TDSplineTDIWaveform : public LISATDIonTheFly{
  public:
    // Orbits *orbits;
    // TDIConfig *tdi_config;
    
    CubicSpline *amp_spline;
    CubicSpline *phase_spline;
    int binary_index_storage;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3){
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
    };
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    // CUDA_DEVICE
    // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    // void run_wave_tdi(
    //     cmplx *tdi_channels_arr, 
    //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
    //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
    // );
    int get_td_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    void check_x();
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);

};

void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


class FDSplineTDIWaveform : public LISATDIonTheFly {
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;
        // double *phase_ref_store;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
    };

    // CUDA_DEVICE
    // FDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_, double *phase_ref_);
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    // CUDA_DEVICE
    // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    // CUDA_DEVICE
    // void run_wave_tdi(
    //     cmplx *tdi_channels_arr, 
    //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
    //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
    // );
    int get_fd_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    double get_phase_ref(double t, double *params, int bin_i);
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);
    CUDA_DEVICE
    void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
    CUDA_DEVICE
    double get_amp_f(double t, double *params, int spline_i);
};

class TFSettings{
  public:
    int num_n;      // number of time pixels/segments
    int num_m;      // number of frequency layers/bins
    int num_channel;
    double df;
    double dt;
    double differential_component;

    // TODO: add to this?
    CUDA_CALLABLE_MEMBER
    TFSettings(double df_, double dt_, int num_m_, int num_n_, int num_channel_, double differential_component_){
        num_m = num_m_;
        num_n = num_n_;
        num_channel = num_channel_;
        df = df_;
        dt = dt_;
        differential_component = differential_component_;
    };
};

/**
 * @brief Generic time-frequency domain container, templated on coefficient type.
 *
 * CoeffT = double  -> WDM (real wavelet coefficients)
 * CoeffT = cmplx   -> STFT (complex Fourier coefficients)
 *
 * All indexing, noise access, inner product accumulation, and TDI-type
 * dispatch are identical for both specializations.  The inner product
 * arithmetic adapts via the overloaded ip_dh / ip_hh helpers in TFTraits.hh.
 *
 * Template method bodies are defined inline (required for CUDA device code
 * templates to be visible at instantiation).
 */
template<typename CoeffT>
class TFDomain : public TFSettings{
  public:
    
    CoeffT *tf_data;    // double* for WDM, cmplx* for STFT
    CoeffT *tf_noise;   // double* for WDM (real PSD), cmplx* for STFT (Hermitian noise matrix)
    int num_data;
    int num_noise;

    CUDA_CALLABLE_MEMBER
    TFDomain(CoeffT *tf_data_, CoeffT *tf_noise_, double df_, double dt_, int num_m_, int num_n_, int num_channel_, int num_data_, int num_noise_, double differential_component_):
    TFSettings(df_, dt_, num_m_, num_n_, num_channel_, differential_component_)
    {
        tf_data = tf_data_;
        tf_noise = tf_noise_;
        num_data = num_data_;
        num_noise = num_noise_;
    };

    // --- Pixel indexing (identical for all CoeffT) ---

    CUDA_DEVICE
    int get_pixel_index(int m, int n, int channel, int data_index)
    {
        if (data_index >= num_data)
        {
#ifdef __CUDACC__  
#else
            throw std::invalid_argument("data_index is larger than available data instances.");
#endif
        }
        return ((data_index * num_channel + channel) * num_m + m) * num_n + n;
    }

    CUDA_DEVICE
    int get_pixel_index_noise(int m, int n, int channel, int noise_index)
    {
        if (noise_index >= num_noise)
        {
#ifdef __CUDACC__  
#else
            throw std::invalid_argument("noise_index is larger than available noise instances.");
#endif
        }
        return ((noise_index * num_channel + channel) * num_m + m) * num_n + n;
    }

    CUDA_DEVICE
    int get_pixel_index_noise_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
    {
        return (((noise_index * num_channel + channel_i) * num_channel + channel_j) * num_m + m) * num_n + n;
    }

    // --- Typed data access ---

    CUDA_DEVICE
    CoeffT get_pixel_data_value(int m, int n, int channel, int data_index)
    {
        return tf_data[get_pixel_index(m, n, channel, data_index)];
    }

    CUDA_DEVICE
    CoeffT get_pixel_noise_value(int m, int n, int channel, int noise_index)
    {
        return tf_noise[get_pixel_index_noise(m, n, channel, noise_index)];
    }

    CUDA_DEVICE
    CoeffT get_pixel_noise_value_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
    {
        return tf_noise[get_pixel_index_noise_cross_channel(m, n, channel_i, channel_j, noise_index)];
    }

    // --- Inner products (use overloaded helpers from TFTraits.hh) ---

    CUDA_DEVICE
    void get_inner_product_value(double *d_h, double *h_h, CoeffT template_nm, int m, int n, int channel, int data_index, int noise_index)
    {
        CoeffT data_nm = get_pixel_data_value(m, n, channel, data_index);
        CoeffT noise_nm = get_pixel_noise_value(m, n, channel, noise_index);
        double val_d_h = ip_dh(data_nm, template_nm, noise_nm, differential_component);
        double val_h_h = ip_hh(template_nm, template_nm, noise_nm, differential_component);
        
        *d_h = val_d_h;
        *h_h = val_h_h;
    }

    CUDA_DEVICE
    void get_inner_product_value_cross_channel(double *d_h, double *h_h, CoeffT template_nm_i, CoeffT template_nm_j, int m, int n, int channel_i, int channel_j, int data_index, int noise_index)
    {
        CoeffT data_nm_i = get_pixel_data_value(m, n, channel_i, data_index);
        CoeffT noise_nm_ij = get_pixel_noise_value_cross_channel(m, n, channel_i, channel_j, noise_index);
        
        double val_d_h = ip_dh(data_nm_i, template_nm_j, noise_nm_ij, differential_component);
        double val_h_h = ip_hh(template_nm_i, template_nm_j, noise_nm_ij, differential_component);
        *d_h = val_d_h;
        *h_h = val_h_h;
    }

    // --- Accumulation with TDI-type dispatch ---

    CUDA_DEVICE
    void add_ip_contrib(double *d_h_tmp, double *h_h_tmp, CoeffT *coeff, int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
#ifdef __CUDACC__
        int tid = threadIdx.x;
#else
        int tid = 0;
#endif

        double d_h_val = 0.0;
        double h_h_val = 0.0;
        if (tdi_type == TDI_XYZ)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                for (int channel_j = 0; channel_j < 3; channel_j += 1)
                {
                    get_inner_product_value_cross_channel(&d_h_val, &h_h_val, coeff[channel_i], coeff[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);                
                    d_h_tmp[tid] += d_h_val;
                    h_h_tmp[tid] += h_h_val;    
                }
            } 
        }
        else if (tdi_type == TDI_AET)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &h_h_val, coeff[channel_i], layer_m, n, channel_i, data_index, noise_index);                
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;    
            } 
        }
        else if (tdi_type == TDI_AE)
        {
            for (int channel_i = 0; channel_i < 2; channel_i += 1)
            {
                get_inner_product_value(&d_h_val, &h_h_val, coeff[channel_i], layer_m, n, channel_i, data_index, noise_index);                
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;    
            } 
        }
    }

    CUDA_DEVICE
    void add_ip_swap_contrib(double *d_h_add_tmp, double *d_h_remove_tmp, double *add_add_tmp, double *remove_remove_tmp, double *add_remove_tmp, CoeffT *coeff_add, CoeffT *coeff_remove, int layer_m, int n, int data_index, int noise_index, int tdi_type)
    {
#ifdef __CUDACC__
        int tid = threadIdx.x;
#else
        int tid = 0;
#endif

        double d_h_add_val = 0.0;
        double d_h_remove_val = 0.0;
        double add_add_val = 0.0;
        double remove_remove_val = 0.0;
        double add_remove_val = 0.0;
        int nchannels = 3;
        if (tdi_type == TDI_AE) nchannels = 2;

        if (tdi_type == TDI_XYZ)
        {
            for (int channel_i = 0; channel_i < 3; channel_i += 1)
            {
                for (int channel_j = 0; channel_j < 3; channel_j += 1)
                {
                    get_inner_product_value_cross_channel(&d_h_add_val, &add_add_val, coeff_add[channel_i], coeff_add[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);                
                    d_h_add_tmp[tid] += d_h_add_val;
                    add_add_tmp[tid] += add_add_val; 
                    
                    get_inner_product_value_cross_channel(&d_h_remove_val, &remove_remove_val, coeff_remove[channel_i], coeff_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);                
                    d_h_remove_tmp[tid] += d_h_remove_val;
                    remove_remove_tmp[tid] += remove_remove_val; 

                    get_inner_product_value_cross_channel(&d_h_remove_val, &add_remove_val, coeff_add[channel_i], coeff_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);                
                    add_remove_tmp[tid] += add_remove_val;
                }
            } 
        }
        else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
        {
            for (int channel_i = 0; channel_i < nchannels; channel_i += 1)
            {
                get_inner_product_value(&d_h_add_val, &add_add_val, coeff_add[channel_i], layer_m, n, channel_i, data_index, noise_index);                
                d_h_add_tmp[tid] += d_h_add_val;
                add_add_tmp[tid] += add_add_val; 
                
                get_inner_product_value(&d_h_remove_val, &remove_remove_val, coeff_remove[channel_i], layer_m, n, channel_i, data_index, noise_index);                
                d_h_remove_tmp[tid] += d_h_remove_val;
                remove_remove_tmp[tid] += remove_remove_val; 

                get_inner_product_value_cross_channel(&d_h_remove_val, &add_remove_val, coeff_add[channel_i], coeff_remove[channel_i], layer_m, n, channel_i, channel_i, data_index, noise_index);                
                add_remove_tmp[tid] += add_remove_val;
            } 
        }
        else
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("Incorrect TDI type.");
#endif
        }
    }
};

// Type aliases for backward compatibility and readability
// NOTE: WDMDomain / STFTDomain are #defined at the top of this file to 
// WDMDomainCPU/GPU and STFTDomainCPU/GPU respectively, so these aliases
// produce the correctly-mangled names for each compilation unit.
typedef TFDomain<double> WDMDomain;
typedef TFDomain<cmplx> STFTDomain;


class WaveletLookupTable : public TFSettings{
  public:
    double *c_nm_all;
    double *s_nm_all;
    
    int num_f;
    int num_fdot;
    double df_interp;
    double dfdot_interp;
    double min_f_scaled;
    double min_fdot;

    CUDA_CALLABLE_MEMBER
    WaveletLookupTable(double *c_nm_all_, double *s_nm_all_, int num_f_, int num_fdot_, double df_interp_, double dfdot_interp_, double min_f_scaled_, double min_fdot_, 
        double df_, double dt_, int num_m_, int num_n_, int num_channel_, double differential_component_): TFSettings(df_, dt_, num_m_, num_n_, num_channel_, differential_component_) {
        // n * num_m + m 
        c_nm_all = c_nm_all_;
        s_nm_all = s_nm_all_;
        num_f = num_f_;
        num_fdot = num_fdot_;
        df_interp = df_interp_;
        dfdot_interp = dfdot_interp_;
        min_f_scaled = min_f_scaled_;
        min_fdot = min_fdot_;
    };
    CUDA_DEVICE
    double linear_interp(double f_scaled, double fdot, double *z_vals);

    /**
     * @brief Project a complex TDI channel value onto a WDM pixel.
     * @return Real wavelet coefficient w_mn = c * Re(h) + s * Im(h).
     */
    CUDA_DEVICE
    double get_coeff(cmplx tdi_channel_val, double f, double fdot, int layer_m);
};


/**
 * @brief STFT lookup table for projecting TDI channel values onto STFT pixels.
 *
 * Stores a precomputed 1D window DFT (complex) that encodes how a
 * monochromatic signal at frequency f maps into STFT bin k.
 * The coefficient is:  h_{nk} = tdi_val * W_hat(k * df - f)
 *
 * The window DFT is stored as a 1D array of complex values, indexed by
 * a uniformly-spaced grid of delta_f = f_bin - f_signal values.
 */
class STFTLookupTable : public TFSettings{
  public:
    cmplx *window_dft;       // precomputed complex window DFT values
    int num_delta_f;          // number of grid points in the window DFT
    double d_delta_f;         // spacing of the delta_f grid
    double min_delta_f;       // minimum delta_f value in the grid
    int window_half_width;    // how many bins of leakage to include on each side

    CUDA_CALLABLE_MEMBER
    STFTLookupTable(cmplx *window_dft_, int num_delta_f_, double d_delta_f_, double min_delta_f_, int window_half_width_,
        double df_, double dt_, int num_m_, int num_n_, int num_channel_, double differential_component_): TFSettings(df_, dt_, num_m_, num_n_, num_channel_, differential_component_) {
        window_dft = window_dft_;
        num_delta_f = num_delta_f_;
        d_delta_f = d_delta_f_;
        min_delta_f = min_delta_f_;
        window_half_width = window_half_width_;
    };

    /**
     * @brief Linearly interpolate the 1D window DFT at a given delta_f.
     */
    CUDA_DEVICE
    cmplx interp_window_dft(double delta_f);

    /**
     * @brief Project a complex TDI channel value onto an STFT pixel.
     * @return Complex STFT coefficient h_{nk} = tdi_val * W_hat(k*df - f).
     */
    CUDA_DEVICE
    cmplx get_coeff(cmplx tdi_channel_val, double f, double fdot, int bin_k);
};

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);

class GBComputationGroup{
  public:
    void gb_wdm_get_ll_wrap(double *d_h_out, double *h_h_out, Orbits* orbits, TDIConfig *tdi_config, WaveletLookupTable* wdm_lookup, TFDomain<double>* wdm, double *params_all, int *data_index_all, int *noise_index_all, int num_bin, int nparams, double T, int tdi_type);
    void gb_stft_get_ll_wrap(double *d_h_out, double *h_h_out, Orbits* orbits, TDIConfig *tdi_config, STFTLookupTable* stft_lookup, TFDomain<cmplx>* stft, double *params_all, int *data_index_all, int *noise_index_all, int num_bin, int nparams, double T, int tdi_type);
};

#endif // __TDI_ON_THE_FLY_HH__