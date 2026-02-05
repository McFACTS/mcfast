use std::f64::consts::PI;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

const M_SUN_KG: f64 = 1.9884099e30;  // Solar mass in kg
const R_SUN_M: f64 = 6.957e8;
const C_SI: f64 = 299792460.0;     // Speed of light in m/s
const G_SI: f64 = 6.67430e-11;     // Gravitational constant in m^3/(kg s^2)
const MPC_SI: f64 = 3.08568e22; // number of meters in a megaparsec
const L_SUN_W: f64 = 3.828e26;       // watts
const YR_S: f64 = 3.15576e7;         // seconds per Julian year

#[pyfunction]
pub fn star_wind_mass_loss_helper<'py>(
    py: Python<'py>,
    disk_star_pro_masses_arr: PyReadonlyArray1<f64>,
    disk_star_pro_log_radius_arr: PyReadonlyArray1<f64>,
    disk_star_pro_log_lum_arr: PyReadonlyArray1<f64>,
    disk_opacity_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64
) -> (Bound<'py, PyArray1<f64>>, f64) {
    let disk_star_pro_masses_slice = disk_star_pro_masses_arr.as_slice().unwrap();
    let disk_star_pro_log_radius_slice = disk_star_pro_log_radius_arr.as_slice().unwrap();
    let disk_star_pro_log_lum_slice = disk_star_pro_log_lum_arr.as_slice().unwrap();
    let disk_opacity_slice = disk_opacity_arr.as_slice().unwrap();

    let star_new_masses_arr = unsafe { PyArray1::new(py, disk_star_pro_masses_slice.len(), false) };
    let star_new_masses_slice = unsafe { star_new_masses_arr.as_slice_mut().unwrap() };

    let timestep_s = timestep_duration_yr * YR_S;

    let mut mass_lost_acc = 0.0_f64; // accumulates in Msun

    for (i, (((star_mass_msun, log_radius), log_lum), disk_opacity)) in disk_star_pro_masses_slice.iter()
        .zip(disk_star_pro_log_radius_slice)
        .zip(disk_star_pro_log_lum_slice)
        .zip(disk_opacity_slice)
        .enumerate()
    {
        // Convert everything to SI up front
        let star_mass_kg = star_mass_msun * M_SUN_KG;
        let star_radius_m = 10.0f64.powf(*log_radius) * R_SUN_M;
        let star_lum_w = 10.0f64.powf(*log_lum) * L_SUN_W;
        // disk_opacity is already in m²/kg

        // Eddington luminosity (watts)
        let l_edd_w = 4.0 * PI * G_SI * C_SI * star_mass_kg / disk_opacity;

        // Escape speed (m/s)
        let v_esc = (2.0 * G_SI * star_mass_kg / star_radius_m).sqrt();

        // Dimensionless tanh argument (both numerator and denominator in watts)
        let tanh_argument = (star_lum_w - l_edd_w) / (0.1 * l_edd_w);

        // Mass loss rate (kg/s), already negative
        let mdot = -(star_lum_w / v_esc.powi(2)) * (1.0 + f64::tanh(tanh_argument));

        // Mass lost this timestep, converted to Msun
        let mass_lost_msun = (mdot * timestep_s) / M_SUN_KG;

        mass_lost_acc += mass_lost_msun;

        let star_new_mass = star_mass_msun + mass_lost_msun;
        debug_assert!(star_new_mass > 0.0, "star_new_mass <= 0 at index {i}");

        star_new_masses_slice[i] = star_new_mass;
    }

    (star_new_masses_arr, mass_lost_acc)
}


// pub fn star_wind_mass_loss_helper<'py>(
//     py: Python<'py>,
//     disk_star_pro_masses_arr: PyReadonlyArray1<f64>,
//     disk_star_pro_log_radius_arr: PyReadonlyArray1<f64>,
//     disk_star_pro_log_lum_arr: PyReadonlyArray1<f64>,
//     disk_opacity_arr: PyReadonlyArray1<f64>,
//     timestep_duration_yr: f64
// ) -> (Bound<'py, PyArray1<f64>>, f64) {
//     let disk_star_pro_masses_slice = disk_star_pro_masses_arr.as_slice().unwrap();
//     let disk_star_pro_log_radius_slice = disk_star_pro_log_radius_arr.as_slice().unwrap();
//     let disk_star_pro_log_lum_slice = disk_star_pro_log_lum_arr.as_slice().unwrap();
//     let disk_opacity_slice = disk_opacity_arr.as_slice().unwrap();
//
//     let star_new_masses_arr = unsafe { PyArray1::new(py, disk_star_pro_masses_slice.len(), false) };
//     let star_new_masses_slice = unsafe { star_new_masses_arr.as_slice_mut().unwrap() };
//
//     // quick and dirty way, manual accumulator
//     let mut mass_lost_acc = 0.0;
//
//     for (i, (((star_mass, disk_star_pro_log_radius), disk_star_pro_log_lum), disk_opacity)) in disk_star_pro_masses_slice.iter()
//         .zip(disk_star_pro_log_radius_slice)
//         .zip(disk_star_pro_log_lum_slice)
//         .zip(disk_opacity_slice)
//         .enumerate() {
//
//         let star_radius = 10.0f64.powf(*disk_star_pro_log_radius); // turn into Rsun??
//         let star_lum = 10.0f64.powf(*disk_star_pro_log_lum); // turn into Lsun
//         // star_mass is in Msun
//         // and noting that timestep_duration_year_si is in years
//         // let timestep_duration_yr_si = //
//
//         // todo: turn to Lsun ???
//         let l_edd = 4.0 * PI * G_SI * C_SI * star_mass / disk_opacity;
//
//         // todo: turn to km/s
//         // star mass in solar masses, star radius is solar radii
//         // let v_esc = (2.0 * G_SI * star_mass / star_radius).sqrt(); 
//         let v_esc = (2.0 * G_SI * (star_mass * M_SUN_KG) / (star_radius * R_SUN_M)).sqrt(); 
//
//         let tanh_argument = (star_lum - l_edd) / (0.1 * l_edd);
//
//         // todo: turn to Msun/yr
//         let mdot_edd = -(star_lum/v_esc.powi(2)) * (1.0 + f64::tanh(tanh_argument));
//
//         // todo: turn to Msun
//         // note that because of mdot_edd, already negative
//         let mass_lost = mdot_edd * timestep_duration_yr;
//
//         mass_lost_acc += mass_lost;
//         let star_new_mass = star_mass + (mdot_edd * timestep_duration_yr);
//
//         star_new_masses_slice[i] = star_new_mass;
//     }
//
//     (star_new_masses_arr, mass_lost_acc)
//
// }

pub fn accrete_star_mass_helper(
    disk_star_pro_masses_arr: PyReadonlyArray1<f64>,
    disk_star_pro_orbs_arr: PyReadonlyArray1<f64>,
    // ???
    disk_star_luminosity_factor: f64,
    // ???
    disk_star_initial_mass_cutoff: f64,
    smbh_mass: f64,
    sound_speed_arr: PyReadonlyArray1<f64>,
    disk_density_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64,
    r_g_in_meters: f64,
) {

}
// def accrete_star_mass(disk_star_pro_masses,
//                       disk_star_pro_orbs_a,
//                       disk_star_luminosity_factor,
//                       disk_star_initial_mass_cutoff,
//                       smbh_mass,
//                       disk_sound_speed,
//                       disk_density,
//                       timestep_duration_yr,
//                       r_g_in_meters):
//     """Adds mass according to Fabj+2024 accretion rate
//
//     Takes initial star masses at start of timestep and adds mass according to Fabj+2024.
//
//     Parameters
//     ----------
//     disk_star_pro_masses : numpy.ndarray
//         Initial masses [M_sun] of stars in prograde orbits around SMBH with :obj:`float` type.
//     disk_star_eddington_ratio : float
//         Accretion rate of fully embedded stars [Eddington accretion rate].
//         1.0=embedded star accreting at Eddington.
//         Super-Eddington accretion rates are permitted.
//         User chosen input set by input file
//     mdisk_star_eddington_mass_growth_rate : float
//         Fractional rate of mass growth AT Eddington accretion rate per year (fixed at 2.3e-8 in mcfacts_sim) [yr^{-1}]
//     timestep_duration_yr : float
//         Length of timestep [yr]
//     r_g_in_meters: float
//         Gravitational radius of the SMBH in meters
//
//     Returns
//     -------
//     disk_star_pro_new_masses : numpy.ndarray
//         Masses [M_sun] of stars after accreting at prescribed rate for one timestep [M_sun] with :obj:`float` type
//
//     Notes
//     -----
//     Calculate Bondi radius: R_B = (2 G M_*)/(c_s **2) and Hill radius: R_Hill \\approx a(1-e)(M_*/(3(M_* + M_SMBH)))^(1/3).
//     Accretion rate is Mdot = (pi/f) * rho * c_s * min[R_B, R_Hill]**2
//     with f ~ 4 as luminosity dependent factor that accounts for the decrease of the accretion rate onto the star as it
//     approaches the Eddington luminosity (see Cantiello+2021), rho as the disk density, and c_s as the sound speed.
//     """
//
//     # Put things in SI units
//     star_masses_si = disk_star_pro_masses * u.solMass
//     disk_sound_speed_si = disk_sound_speed(disk_star_pro_orbs_a) * u.meter/u.second
//     disk_density_si = disk_density(disk_star_pro_orbs_a) * (u.kg / (u.m ** 3))
//     timestep_duration_yr_si = timestep_duration_yr * u.year
//
//     # Calculate Bondi and Hill radii
//     r_bondi = (2 * const.G.to("m^3 / kg s^2") * star_masses_si / (disk_sound_speed_si ** 2)).to("meter")
//     r_hill_rg = (disk_star_pro_orbs_a * ((disk_star_pro_masses / (3 * (disk_star_pro_masses + smbh_mass))) ** (1./3.)))
//     r_hill_m = si_from_r_g(smbh_mass, r_hill_rg, r_g_defined=r_g_in_meters)
//
//     # Determine which is smaller for each star
//     min_radius = np.minimum(r_bondi, r_hill_m)
//
//     # Calculate the mass accretion rate
//     mdot = ((np.pi / disk_star_luminosity_factor) * disk_density_si * disk_sound_speed_si * (min_radius ** 2)).to("kg/yr")
//
//     # Accrete mass onto stars
//     disk_star_pro_new_masses = ((star_masses_si + mdot * timestep_duration_yr_si).to("Msun")).value
//
//     # Stars can't accrete over disk_star_initial_mass_cutoff
//     disk_star_pro_new_masses[disk_star_pro_new_masses > disk_star_initial_mass_cutoff] = disk_star_initial_mass_cutoff
//
//     # Mass gained does not include the cutoff
//     mass_gained = ((mdot * timestep_duration_yr_si).to("Msun")).value
//
//     # Immortal stars don't enter this function as immortal because they lose a small amt of mass in star_wind_mass_loss
//     # Get how much mass is req to make them immortal again
//     immortal_mass_diff = disk_star_pro_new_masses[disk_star_pro_new_masses == disk_star_initial_mass_cutoff] - disk_star_pro_masses[disk_star_pro_new_masses == disk_star_initial_mass_cutoff]
//     # Any extra mass over the immortal cutoff is blown off the star and back into the disk
//     immortal_mass_lost = mass_gained[disk_star_pro_new_masses == disk_star_initial_mass_cutoff] - immortal_mass_diff
//
//     assert np.all(disk_star_pro_new_masses > 0), \
//         "disk_star_pro_new_masses has values <= 0"
//
//     return disk_star_pro_new_masses, mass_gained.sum(), immortal_mass_lost.sum()
