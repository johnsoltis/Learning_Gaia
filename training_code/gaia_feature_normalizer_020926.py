import numpy as np

#Don't really care about all the sources with par~0, want to make sure I get the nearby stuff right though!
#Similarly, larger pm's are more likely to correspond to nearby objects, where presumably the measurement is more correct
#features with the n_obs format are all integers running from 0 to N, lending themselves to min_max normalization readily
P_Z_SCORE = ['parallax', 'pmra', 'pmdec', 'pml', 'pmb', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'radial_velocity', 'bp_rp', 'phot_bp_rp_excess_factor']
MIN_MAX = ['logg_gspphot','logg_gspphot_lower','logg_gspphot_upper','mh_gspphot','mh_gspphot_lower','mh_gspphot_upper','nu_eff_used_in_astrometry','pseudocolour','pseudocolour_error','phot_bp_n_contaminated_transits','phot_bp_n_blended_transits','phot_rp_n_contaminated_transits','phot_rp_n_blended_transits','phot_proc_mode','rv_method_used','rv_nb_transits','rv_nb_deblended_transits','rv_visibility_periods_used','rv_renormalised_gof','rv_template_logg','rv_template_fe_h','rv_atm_param_origin','vbroad_nb_transits','grvs_mag','grvs_mag_nb_transits','grvs_mag_nb_transits']

LOG_1_MAX = ['astrometric_n_obs_ac', 'astrometric_n_good_obs_al','astrometric_n_bad_obs_al', 'matched_transits_removed', 'astrometric_excess_noise', 'astrometric_excess_noise_sig', 'ipd_gof_harmonic_amplitude', 'rv_expected_sig_to_noise', 'rv_time_duration', 'rv_amplitude_robust', 'vbroad', 'vbroad_error']

class FormatError(Exception):
    """Exception raised for normalization."""
    def __init__(self, message="A id shard loading error occurred."):
        self.message = message
        super().__init__(self.message)


def l_stepwise_transform(x,y):
    xy_norm = np.sqrt(x**2 + y**2)
    x_hat = x/xy_norm
    y_hat = y/xy_norm
    transform_z = np.empty(len(x))
    transform_z[(y_hat >= 0)] = (180*(np.arccos(x_hat[(y_hat >= 0)]))/np.pi)
    transform_z[(y_hat < 0)] =  360 - (180*(np.arccos(x_hat[(y_hat < 0)]))/np.pi) #(180*(np.arccos(y[(x < 0)]))/np.pi) - (np.arcsin(x[(x < 0)])/np.pi)
    return transform_z


#normalizes all data approximately that it falls between 0 and 1
def feature_normer(data_column, feature_name, normalization_dictionary, debug_file_name = ''):
    if len(data_column) < 1:
        raise FormatError(f"Feature {feature_name} column is empty. From file {debug_file_name}.")

    if feature_name == 'ra' or feature_name == 'ecl_lon':
        return (data_column/360)
    
    #Note that if the feature is l, the normalized version is split between two variables, sin(l) and cos(l)
    elif feature_name == 'l':
        return np.cos(np.pi * data_column/180), np.sin(np.pi * data_column/180)
        
    elif feature_name == 'b':
        return data_column/90 #np.sin(np.pi * data_column/180)
        
    elif feature_name in P_Z_SCORE:
        return (data_column - normalization_dictionary[feature_name]['est_median'])/(normalization_dictionary[feature_name]['est_p97_5'] - normalization_dictionary[feature_name]['est_p2_5'])
    
    elif feature_name == 'ra_error' or feature_name == 'dec_error':
        return (data_column/100)
    
    elif feature_name == 'dec' or feature_name == 'ecl_lat' or feature_name == 'scan_direction_mean_k2':
        return (data_column/180) + 0.5

    elif feature_name == 'astrometric_gof_al':
        if np.min(data_column) + 100 < 0:
            print(f"Feature {feature_name} has negative values in log. MIN = {np.min(data_column)}. log(x + 100) norm.")
            raise FormatError(f"Feature {feature_name} has negative values in log. MIN = {np.min(data_column)}. log(x + 100) norm.")
        return np.log10(data_column + 100)/np.log10(normalization_dictionary[feature_name]['max'] + 100)
        
    elif feature_name == 'parallax_error':
        return data_column/6
        
    elif feature_name == 'pmra_error' or feature_name == 'pmdec_error':
        return data_column/3.5
        
    elif feature_name in MIN_MAX:
        return (data_column - normalization_dictionary[feature_name]['min']) / (normalization_dictionary[feature_name]['max'] - normalization_dictionary[feature_name]['min'])
    
    elif feature_name[-4:] == 'corr':
        return (data_column/2 + .5)
    
    elif feature_name[-6:] == 'gof_al' or feature_name[-7:] == 'chi2_al' or feature_name == 'astrometric_sigma5d_max' or feature_name[-16:] == 'matched_transits' or feature_name[-16:] == 'matched_transits' or feature_name[-5:] == 'n_obs' or feature_name[-10:] == 'flux_error' or feature_name in LOG_1_MAX:
        if np.min(data_column) < 0:
            print(f"Feature {feature_name} has negative values in log. MIN = {np.min(data_column)}. log+1 norm.")
            raise FormatError(f"Feature {feature_name} has negative values in log. MIN = {np.min(data_column)}. log+1 norm.")
        if np.log10(normalization_dictionary[feature_name]['max'] + 1) < 1e-16:
            print(f"Feature {feature_name} has too small of max value, will cause overflow. log+1 norm.")
            raise FormatError(f"Feature {feature_name} has too small of max value, will cause overflow. log+1 norm.")
        return np.log10(data_column + 1)/np.log10(normalization_dictionary[feature_name]['max'] + 1)
    
    elif feature_name[:3] == 'in_' or feature_name[:4] == 'has_' or feature_name[:9] == 'classprob' or feature_name == 'astrometric_primary_flag' or feature_name[:23] == 'scan_direction_strength' or feature_name == 'duplicated_source' or feature_name == 'rv_chisq_pvalue':
        return data_column
    
    elif feature_name[:4] == 'teff' or feature_name[:8] == 'distance' or feature_name == 'visibility_periods_used' or feature_name == 'rv_template_teff' or feature_name == 'rvs_spec_sig_to_noise':
        if np.min(data_column/normalization_dictionary[feature_name]['min']) < 0:
            print(f"Feature {feature_name} has negative values in log. MIN = {np.min(data_column)}. log(x/min) norm.")
            raise FormatError(f"Feature {feature_name} has negative values in log. MIN = {np.min(data_column)}. log(x/min) norm.")
        if np.log10(normalization_dictionary[feature_name]['max']/normalization_dictionary[feature_name]['min']) < 1e-16:
            print(f"Feature {feature_name} has too small of a max/min value, will cause overflow in division. log(x/min) norm.")
            raise FormatError(f"Feature {feature_name} has too small of a max/min value, will cause overflow in division. log(x/min) norm.")
            
        return np.log10(data_column/normalization_dictionary[feature_name]['min'])/np.log10(normalization_dictionary[feature_name]['max']/normalization_dictionary[feature_name]['min'])
    
    elif feature_name[:5] == 'azero':
        return data_column/10
    
    elif feature_name[:3] == 'ag_':
        return data_column/7.5
    
    elif feature_name[:9] == 'ebpminrp_':
        return data_column/7.5
    
    elif feature_name == 'astrometric_params_solved' or feature_name == 'ipd_frac_multi_peak' or feature_name == 'ipd_frac_odd_win' or feature_name == 'ruwe':
        return data_column/100

    elif feature_name == 'ipd_gof_harmonic_phase':
        return data_column/180
        
    elif feature_name == 'scan_direction_mean_k1':
        return (data_column/360) + 0.5
        
    elif feature_name == 'scan_direction_mean_k3':
        return (data_column/120) + 0.5

    elif feature_name == 'scan_direction_mean_k4':
        return (data_column/90) + 0.5
        
    elif feature_name == 'radial_velocity_error':
        return data_column/40
        
    elif feature_name == 'phot_variable_flag' or feature_name == 'grvs_mag_error':
        return data_column/2
    
    elif feature_name == 'non_single_star':
        return data_column/7
        
    else:
        print(f"Function missing feature {feature_name}.")
        raise FormatError(f"Function missing feature {feature_name}.")
        
def feature_denormer(normed_data, feature_name, normalization_dictionary):
    if feature_name == 'ra' or feature_name == 'ecl_lon':
        return normed_data * 360
        
    elif feature_name == 'l':
        l_1 = normed_data[0]
        l_2 = normed_data[1]
        return l_stepwise_transform(l_1, l_2)
        
    elif feature_name == 'b':
        return normed_data * 90#180*(np.arcsin(normed_data)/np.pi)
        
    elif feature_name in P_Z_SCORE:
        return (normed_data*(normalization_dictionary[feature_name]['est_p97_5'] - normalization_dictionary[feature_name]['est_p2_5'])) + normalization_dictionary[feature_name]['est_median']
        
    elif feature_name == 'ra_error' or feature_name == 'dec_error' or feature_name == 'astrometric_params_solved' or feature_name == 'ipd_frac_multi_peak' or feature_name == 'ipd_frac_odd_win' or feature_name == 'ruwe':
        return normed_data * 100
    
    elif feature_name == 'dec' or feature_name == 'ecl_lat' or feature_name == 'scan_direction_mean_k2':
        return (normed_data - 0.5) * 180
        
    elif feature_name == 'parallax_error':
        return normed_data * 6
        
    #elif feature_name == 'pmra' or feature_name == 'pmdec':
    #    return inverse_pm_norm(normed_data)#((normalization_dictionary[feature_name]['max'] + 10000)**normed_data) - 10000
        
    elif feature_name == 'pmra_error' or feature_name == 'pmdec_error':
        return normed_data * 3.5
        
    elif feature_name in MIN_MAX:
        return (normed_data * (normalization_dictionary[feature_name]['max'] - normalization_dictionary[feature_name]['min'])) + normalization_dictionary[feature_name]['min']
    
    elif feature_name[-4:] == 'corr':
        return (normed_data - 0.5) * 2
    
    elif feature_name[-6:] == 'gof_al' or feature_name[-7:] == 'chi2_al' or feature_name == 'astrometric_sigma5d_max' or feature_name[-16:] == 'matched_transits' or feature_name[-16:] == 'matched_transits' or feature_name[-5:] == 'n_obs' or feature_name[-10:] == 'flux_error' or feature_name in LOG_1_MAX:
        return ((normalization_dictionary[feature_name]['max'] + 1)**normed_data) - 1
    
    elif feature_name[:3] == 'in_' or feature_name[:4] == 'has_' or feature_name[:9] == 'classprob' or feature_name == 'astrometric_primary_flag' or feature_name[:23] == 'scan_direction_strength' or feature_name == 'duplicated_source' or feature_name == 'rv_chisq_pvalue':
        return normed_data
    
    elif feature_name[:4] == 'teff' or feature_name[:8] == 'distance' or feature_name == 'visibility_periods_used' or feature_name == 'rv_template_teff' or feature_name == 'rvs_spec_sig_to_noise':
        return normalization_dictionary[feature_name]['min'] * (normalization_dictionary[feature_name]['max']/normalization_dictionary[feature_name]['min'])**(normed_data)
    
    elif feature_name[:5] == 'azero':
        return normed_data * 10
    
    elif feature_name[:3] == 'ag_' or feature_name[:9] == 'ebpminrp_':
        return normed_data * 7.5

    elif feature_name == 'ipd_gof_harmonic_phase':
        return normed_data * 180
        
    elif feature_name == 'scan_direction_mean_k1':
        return (normed_data - 0.5) * 360
        
    elif feature_name == 'scan_direction_mean_k3':
        return (normed_data - 0.5) * 120

    elif feature_name == 'scan_direction_mean_k4':
        return (normed_data - 0.5) * 90
        
    elif feature_name == 'radial_velocity_error':
        return normed_data * 40
        
    elif feature_name == 'phot_variable_flag' or feature_name == 'grvs_mag_error':
        return normed_data * 2
    
    elif feature_name == 'non_single_star':
        return normed_data * 7
        
    else:
        print(f"Function missing feature {feature_name}.")
        raise FormatError(f"Function missing feature {feature_name}.")

