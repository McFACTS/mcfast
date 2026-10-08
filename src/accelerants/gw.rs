use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use std::f64::consts::PI;
use crate::accelerants::{C_SI, FloatArray1, G_SI, M_SUN_KG, MPC_SI, units::{self, si_from_r_g}};

#[pyfunction]
pub fn gw_hardening_helper<'py>(
    py: Python<'py>,
    mass_1_arr: PyReadonlyArray1<f64>,
    mass_2_arr: PyReadonlyArray1<f64>,
    bin_ecc_arr: PyReadonlyArray1<f64>,
    bin_sep_arr: PyReadonlyArray1<f64>,
    bin_time_to_merge_arr: PyReadonlyArray1<f64>,
    flag_merging_arr: PyReadonlyArray1<f64>,
    smbh_mass: f64,
    timestep_length: f64,
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>, FloatArray1<'py>)> {

    let mass_1_slice = mass_1_arr.as_slice().unwrap();
    let mass_2_slice = mass_2_arr.as_slice().unwrap();
    let bin_ecc_slice = bin_ecc_arr.as_slice().unwrap();
    let bin_sep_slice = bin_sep_arr.as_slice().unwrap();
    let bin_time_to_merge_slice = bin_time_to_merge_arr.as_slice().unwrap();
    let flag_merging_slice = flag_merging_arr.as_slice().unwrap();

    let array_len = mass_1_slice.len();

    let new_time_to_merge_arr = unsafe { PyArray1::new(py, array_len, false) };
    let new_time_to_merge_slice = unsafe { new_time_to_merge_arr.as_slice_mut().unwrap() };

    let new_bin_sep_arr = unsafe { PyArray1::new(py, array_len, false) };
    let new_bin_sep_slice = unsafe { new_bin_sep_arr.as_slice_mut().unwrap() };

    let new_flag_merging_arr = unsafe { PyArray1::new(py, array_len, false) };
    let new_flag_merging_slice = unsafe { new_flag_merging_arr.as_slice_mut().unwrap() };

    // from years to seconds
    let timestep_duration_sec = timestep_length * 31557600.0;

    for (i, (((((m1, m2), ecc), sep), time_to_merge), flag_merging)) in mass_1_slice.iter()
        .zip(mass_2_slice)
        .zip(bin_ecc_slice)
        .zip(bin_sep_slice)
        .zip(bin_time_to_merge_slice)
        .zip(flag_merging_slice)
        .enumerate() {

        let flag_not_merging = *flag_merging >= 0.0;

        // may be fine to have this if statement, since it's allowing us to 
        // skip a bunch of non-vectorizable calls
        let (time_to_merger_gw, sep_crit) = if flag_not_merging {
            let ecc_factor_1 = (1.0 - ecc.powi(2)).powf(3.5);
            let ecc_factor_2 = 1.0 + ((73.0/24.0) * ecc.powi(2)) + ((37.0/96.0) * ecc.powi(4));
            let ecc_factor = ecc_factor_1/ecc_factor_2;

            // safe to combine these and call once, since r_schwarzschild_of_m is commutative
            let sep_crit = units::r_schwarzschild_of_m_local(m1 + m2);

            let time_to_merger_gw = time_of_orbital_shrinkage(*m1, *m2, si_from_r_g(smbh_mass, *sep), sep_crit) * ecc_factor;
            (time_to_merger_gw, sep_crit)
        } else {
            // NOTE: sep_crit may be used again for new_bin_sep, but 
            // only when flag_not_merging, so it's fine to pass a placeholder here
            (*time_to_merge, 0.0)
        };

        assert!(time_to_merger_gw.is_finite());

        new_time_to_merge_slice[i] = time_to_merger_gw;

        let is_merging = time_to_merger_gw <= timestep_duration_sec;

        new_bin_sep_slice[i] = if !flag_not_merging || !is_merging {
            *sep
        } else if is_merging {
           units::r_g_from_units(smbh_mass, sep_crit) 
        } else {
            0.0
        };

        new_flag_merging_slice[i] = if !flag_not_merging || !is_merging {
            *flag_merging
        } else if is_merging {
            // have to make a rust-local version of rg_from_units? didn't have one before ig
           -2.0
        } else {
            0.0
        };
    }
    Ok((new_bin_sep_arr, new_time_to_merge_arr, new_flag_merging_arr))
}


// def gw_hardening(mass_1, mass_2, bin_ecc, bin_sep, bin_time_to_merge, flag_merging, smbh_mass, timestep_length, r_g_in_meters):
//     array_length = len(mass_1)
//
//     flag_not_merging = np.array([flag_merging[i] >= 0 for i in range(array_length)], dtype=np.bool_)
//
//     # Find eccentricity factor (1-e_b^2)^7/2
//     ecc_factor_1 = np.power(1 - np.power(bin_ecc[flag_not_merging], 2), 3.5)
//     # and eccentricity factor [1+(73/24)e_b^2+(37/96)e_b^4]
//     ecc_factor_2 = 1 + ((73 / 24) * np.power(bin_ecc[flag_not_merging], 2)) + (
//                 (37 / 96) * np.power(bin_ecc[flag_not_merging], 4))
//     # overall ecc factor = ecc_factor_1/ecc_factor_2
//     ecc_factor = ecc_factor_1 / ecc_factor_2
//
//     # sep_crit = (unit_conversion.r_schwarzschild_of_m(mass_1) +
//     #             unit_conversion.r_schwarzschild_of_m(mass_2))
//     sep_crit = (unit_conversion.r_schwarzschild_of_m_optimized(mass_1+mass_2))
//
//     time_to_merger_gw = (peters.time_of_orbital_shrinkage(
//         mass_1[flag_not_merging] * u.Msun,
//         mass_2[flag_not_merging] * u.Msun,
//         unit_conversion.si_from_r_g_optimized(smbh_mass, bin_sep[flag_not_merging], r_g_defined=r_g_in_meters),
//         sep_final=sep_crit[flag_not_merging]
//     ) * ecc_factor).value
//
//     assert np.isfinite(time_to_merger_gw).all(), \
//         "Finite check failure: time_to_merger_gw"
//
//     new_time_to_merge = np.zeros(array_length)
//     new_time_to_merge[~flag_not_merging] = bin_time_to_merge[~flag_not_merging]
//     new_time_to_merge[flag_not_merging] = time_to_merger_gw
//
//     timestep_duration_sec = (timestep_length * u.yr).to("second").value
//     merge_mask = new_time_to_merge <= timestep_duration_sec
//
//     new_bin_sep = np.zeros(array_length)
//     new_bin_sep[~flag_not_merging] = bin_sep[~flag_not_merging]
//     new_bin_sep[~merge_mask] = bin_sep[~merge_mask]
//     new_bin_sep[merge_mask] = unit_conversion.r_g_from_units_optimized(smbh_mass, sep_crit[merge_mask])
//
//     new_flag_merging = np.zeros(array_length, dtype=np.int_)
//     new_flag_merging[~flag_not_merging] = flag_merging[~flag_not_merging]
//     new_flag_merging[~merge_mask]= flag_merging[~merge_mask]
//     new_flag_merging[merge_mask] = -2
//
//     return new_bin_sep, new_time_to_merge, new_flag_merging
//






// scalar peters time of orbital shrinkage helper
fn time_of_orbital_shrinkage(m1: f64, m2: f64, sep_initial: f64, sep_final: f64) -> f64 {
    // taking these in as solar masses, turning to kg
    let mass1 = m1 * M_SUN_KG;
    let mass2 = m2 * M_SUN_KG;

    // these should already be in meters?? 
    //  sep_initial = sep_initial.to(u.m).value
    //  sep_final = sep_final.to(u.m).value

    // powi is non-const, so we can't set it up as a constant value
    let g_c: f64 = ((64.0 / 5.0) * (G_SI.powi(3))) * (C_SI.powi(-5));
    let beta = g_c * mass1 * mass2 * (mass1 + mass2);
    let time_of_shrinkage = ((sep_initial.powi(4)) - (sep_final.powi(4))) / 4.0 / beta;

    debug_assert!(time_of_shrinkage >= 0.0);

    // unit in seconds
    time_of_shrinkage
}


#[pyfunction(signature=(smbh_mass, disk_bh_pro_orbs_a_arr, disk_bh_pro_masses_arr, disk_bh_pro_orbs_ecc_arr, timestep_duration_yr, inner_disk_outer_radius, disk_inner_stable_circ_orb))]
pub fn bh_near_smbh<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    disk_bh_pro_orbs_a_arr: PyReadonlyArray1<f64>,
    disk_bh_pro_masses_arr: PyReadonlyArray1<f64>,
    disk_bh_pro_orbs_ecc_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64,
    inner_disk_outer_radius: f64,
    disk_inner_stable_circ_orb: f64,
) -> PyResult<FloatArray1<'py>> {

    let disk_bh_pro_orbs_a_slice = disk_bh_pro_orbs_a_arr.as_slice().unwrap();
    let disk_bh_pro_masses_slice = disk_bh_pro_masses_arr.as_slice().unwrap();
    // not currently used, but may be used in the future
    let _disk_bh_pro_orbs_ecc_slice = disk_bh_pro_orbs_ecc_arr.as_slice().unwrap();

    let new_disk_bh_pro_orbs_a_arr = unsafe { PyArray1::new(py, disk_bh_pro_orbs_a_slice.len(), false) };
    let new_disk_bh_pro_orbs_a_slice = unsafe { new_disk_bh_pro_orbs_a_arr.as_slice_mut().unwrap() };

    // minimum safe distance in r_g
    let min_safe_distance = disk_inner_stable_circ_orb.max(inner_disk_outer_radius);

    // for (i, ((orb_a, mass), ecc)) in disk_bh_pro_orbs_a_slice.iter()
    for (i, (orb_a, mass)) in disk_bh_pro_orbs_a_slice.iter()
        .zip(disk_bh_pro_masses_slice)
        // .zip(disk_bh_pro_orbs_ecc_slice)
        .enumerate() {

        let new_loc = if *orb_a < min_safe_distance {

            // not currently used, but may be added later
            // let ecc_factor_arr = (1.0 - (ecc).powf(2.0)).powf(7.0/2.0);

            // time_of_orbital_shrinkage returns seconds, turn it into years
            let decay_timesteps = time_of_orbital_shrinkage(
                smbh_mass, 
                *mass, 
                si_from_r_g(smbh_mass, *orb_a), 
                0.0
            ) * 31557600.0 / timestep_duration_yr;

            // in cases where decay_timesteps is 0, clamp decrement to 0
            // more elegant way to do it?
            let decrement = if decay_timesteps == 0.0 {
                0.0
            } else {
                1.0 - (1.0/decay_timesteps)
            };

            (decrement * orb_a).clamp(1.0, f64::INFINITY)
        } else {
            *orb_a
        };

        new_disk_bh_pro_orbs_a_slice[i] = new_loc;
    }

    Ok(new_disk_bh_pro_orbs_a_arr)
}


#[pyfunction(signature=(mass_1_obj, mass_2_arr, obj_sep_arr, timestep_duration_yr, old_gw_freq_arr, smbh_mass, agn_redshift, flag_include_old_gw_freq))]
pub fn gw_strain_helper<'py>(
    py: Python<'py>,
    mass_1_obj: &Bound<'_, PyAny>, // PyReadonlyArray1<f64>,  
    mass_2_arr: PyReadonlyArray1<f64>,
    obj_sep_arr: PyReadonlyArray1<f64>, 
    timestep_duration_yr: f64, 
    old_gw_freq_arr: PyReadonlyArray1<f64>, 
    smbh_mass: f64, 
    agn_redshift: f64,
    flag_include_old_gw_freq: bool, // defaults to true?
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>, FloatArray1<'py>)> {

    let mass_2_slice = mass_2_arr.as_slice().unwrap();
    let obj_sep_slice = obj_sep_arr.as_slice().unwrap();
    let old_gw_freq_slice = old_gw_freq_arr.as_slice().unwrap();

    let char_strain_arr = unsafe { PyArray1::new(py, mass_2_slice.len(), false) };
    let char_strain_slice = unsafe { char_strain_arr.as_slice_mut().unwrap() };

    let strain_arr = unsafe { PyArray1::new(py, mass_2_slice.len(), false) };
    let strain_slice = unsafe { strain_arr.as_slice_mut().unwrap() };

    let nu_gw_arr = unsafe { PyArray1::new(py, mass_2_slice.len(), false) };
    let nu_gw_slice = unsafe { nu_gw_arr.as_slice_mut().unwrap() };

    // turn years into seconds
    // not quite 365*24*60*60, slightly higher
    let timestep_units = timestep_duration_yr * 31557600.0;

    // rg is in meters
    let rg = 1.5e11 * (smbh_mass/1e8f64);

    if let Ok(mass_1_arr) = mass_1_obj.extract::<PyReadonlyArray1<f64>>() {
        let mass_1_slice = mass_1_arr.as_slice().unwrap();

        for (i, (((mass_1, mass_2), obj_sep), old_gw_freq)) in mass_1_slice.iter()
            .zip(mass_2_slice)
            .zip(obj_sep_slice)
            .zip(old_gw_freq_slice)
            .enumerate() {

            // cds Msun is just 1.98840987e+30 kg, same as M_SUN_KG
            let mass_1 = mass_1 * M_SUN_KG;
            let mass_2 = mass_2 * M_SUN_KG;

            let mass_total = mass_1 + mass_2;

            let bin_sep = obj_sep * rg;

            let mass_chirp = ((mass_1 * mass_2).powf(3.0/5.0)) / (mass_total.powf(1.0/5.0));

            // already in meters
            let rg_chirp = (G_SI * mass_chirp) / C_SI.powi(2);

            let bin_sep = bin_sep.max(rg_chirp);

            // already in Hz
            let nu_gw = (1.0 / PI) * (mass_total * G_SI / bin_sep.powi(3)).sqrt();

            let d_obs = match agn_redshift {
                0.1 => 421.0 * MPC_SI,
                0.5 => 1909.0 * MPC_SI,
                _ => panic!("The only valid values for agn_redshift are {{0.1, 0.5}}, not {}", agn_redshift),
            };

            let strain = (4.0/d_obs) * rg_chirp * (PI * nu_gw * rg_chirp / C_SI).powf(2.0/3.0);

            // gwb isn't used in the original code, can take it out entirely
            // let gwb = nu_gw > 2.0e-3f64;

            let tight = nu_gw > 2e-3;
            let lessonesix = nu_gw < 1e-6;
            let greateronesix = nu_gw > 1e-6;

            // should precisely match the current logic of the strain_factor creation, though
            // the fallthrough for flag false and nu_gw == 1e-6 is inelegant and may be revised
            let strain_factor = match (flag_include_old_gw_freq, tight, lessonesix, greateronesix) {
                (true, false, _, _) => {
                    let delta_nu = (old_gw_freq - nu_gw).abs();
                    let delta_nu_delta_timestep = delta_nu / timestep_units;
                    let nu_squared = nu_gw.powi(2);
                    // nu_factor is also not used in the original code
                    // let nu_factor = nu_gw.powf(-5.0/6.0);

                    ((nu_squared / delta_nu_delta_timestep) / 8.0).sqrt()
                },
                (true, true, _, _) => {
                    let num_factor = (5.0f64/96.0f64).sqrt() * (1.0/(8.0*PI)) * (1.0 / PI).powf(1.0/3.0);
                    num_factor * ((C_SI / rg_chirp).powf(5.0/6.0)) * (nu_gw).powf(-5.0/6.0)
                },
                (false, _, false, true) => 4.0e3,
                (false, _, true, false) => {
                    (nu_gw * PI * 1e7 / 8.0).sqrt()
                }
                _ => 1.0
            };

            let char_strain = strain_factor*strain;

            char_strain_slice[i] = char_strain;
            strain_slice[i] = strain;
            nu_gw_slice[i] = nu_gw;
        }
        Ok((char_strain_arr, strain_arr, nu_gw_arr))

    } else if let Ok(mass_1) = mass_1_obj.extract::<f64>() {

        for (i, ((mass_2, obj_sep), old_gw_freq)) in mass_2_slice.iter()
            .zip(obj_sep_slice)
            .zip(old_gw_freq_slice)
            .enumerate() {

            // cds Msun is just 1.98840987e+30 kg, same as M_SUN_KG
            let mass_1 = mass_1 * M_SUN_KG;
            let mass_2 = mass_2 * M_SUN_KG;

            let mass_total = mass_1 + mass_2;

            let bin_sep = obj_sep * rg;

            let mass_chirp = ((mass_1 * mass_2).powf(3.0/5.0)) / (mass_total.powf(1.0/5.0));

            // already in meters
            let rg_chirp = (G_SI * mass_chirp) / C_SI.powi(2);

            let bin_sep = bin_sep.max(rg_chirp);

            // already in Hz
            let nu_gw = (1.0 / PI) * (mass_total * G_SI / bin_sep.powi(3)).sqrt();

            let d_obs = match agn_redshift {
                0.1 => 421.0 * MPC_SI,
                0.5 => 1909.0 * MPC_SI,
                _ => panic!("The only valid values for agn_redshift are {{0.1, 0.5}}, not {}", agn_redshift),
            };

            let strain = (4.0/d_obs) * rg_chirp * (PI * nu_gw * rg_chirp / C_SI).powf(2.0/3.0);

            // gwb isn't used in the original code, can take it out entirely
            // let gwb = nu_gw > 2.0e-3f64;

            let tight = nu_gw > 2e-3;
            let lessonesix = nu_gw < 1e-6;
            let greateronesix = nu_gw > 1e-6;

            // should precisely match the current logic of the strain_factor creation, though
            // the fallthrough for flag false and nu_gw == 1e-6 is inelegant and may be revised
            let strain_factor = match (flag_include_old_gw_freq, tight, lessonesix, greateronesix) {
                (true, false, _, _) => {
                    let delta_nu = (old_gw_freq - nu_gw).abs();
                    let delta_nu_delta_timestep = delta_nu / timestep_units;
                    let nu_squared = nu_gw.powi(2);
                    // nu_factor is also not used in the original code
                    // let nu_factor = nu_gw.powf(-5.0/6.0);

                    ((nu_squared / delta_nu_delta_timestep) / 8.0).sqrt()
                },
                (true, true, _, _) => {
                    let num_factor = (5.0f64/96.0f64).sqrt() * (1.0/(8.0*PI)) * (1.0 / PI).powf(1.0/3.0);
                    num_factor * ((C_SI / rg_chirp).powf(5.0/6.0)) * (nu_gw).powf(-5.0/6.0)
                },
                (false, _, false, true) => 4.0e3,
                (false, _, true, false) => {
                    (nu_gw * PI * 1e7 / 8.0).sqrt()
                }
                _ => 1.0
            };

            let char_strain = strain_factor*strain;

            char_strain_slice[i] = char_strain;
            strain_slice[i] = strain;
            nu_gw_slice[i] = nu_gw;
        }
        Ok((char_strain_arr, strain_arr, nu_gw_arr))

    } else {
        Err(PyValueError::new_err("Input `retro_mass derived from retrograde_bh_masses is neither a numeric scalar nor a numpy ndarray."))
        
    }
}
