#include "sgp4_propagator.h"
#include "SGP4.h"
#include "Tle.h"
#include "Eci.h"
#include "DateTime.h"
#include <random>
#include <string>
#include <Eigen/Dense>
#include <astro/orbitalElementConversions.hpp>

SGP4Propagator::SGP4Propagator(const Eigen::MatrixXd& process_noise_covariance)
    : process_noise_covariance_(process_noise_covariance) {}

Particle SGP4Propagator::propagate(const Particle& particle, double dt) const {
    Particle propagated_particle = particle;
    // Extract ECI position and velocity from state_vector
    Eigen::Vector3d pos_m = particle.state_vector.head(3); // meters
    Eigen::Vector3d vel_ms = particle.state_vector.segment(3, 3); // m/s
    // Constants
    const double pi = 3.14159265358979323846;
    const double mu_km = 398600.4418; // km^3/s^2
    // Convert to km and km/s for TLE/SGP4
    Eigen::Vector3d pos_km = pos_m / 1000.0;
    Eigen::Vector3d vel_kms = vel_ms / 1000.0;
    // Compose 6D vector for OpenAstro
    Eigen::Matrix<double, 6, 1> cartesian_km;
    cartesian_km << pos_km, vel_kms;
    // Convert to Keplerian elements (in km, km/s)
    auto kep = astro::convertCartesianToKeplerianElements(cartesian_km, mu_km);
    // Extract elements using OpenAstro indices
    double sma = kep[astro::semiMajorAxisIndex]; // km
    double ecc = kep[astro::eccentricityIndex];
    double inc_deg = kep[astro::inclinationIndex] * 180.0 / pi;
    double raan_deg = kep[astro::longitudeOfAscendingNodeIndex] * 180.0 / pi;
    double argp_deg = kep[astro::argumentOfPeriapsisIndex] * 180.0 / pi;
    double true_anom = kep[astro::trueAnomalyIndex]; // rad
    // Convert true anomaly to mean anomaly for TLE
    double mean_anom = 0.0;
    if (ecc < 1.0) {
        double ecc_anom = astro::convertTrueAnomalyToEllipticalEccentricAnomaly(true_anom, ecc);
        mean_anom = astro::convertEllipticalEccentricAnomalyToMeanAnomaly(ecc_anom, ecc);
    } else if (ecc > 1.0) {
        double ecc_anom = astro::convertTrueAnomalyToHyperbolicEccentricAnomaly(true_anom, ecc);
        mean_anom = astro::convertHyperbolicEccentricAnomalyToMeanAnomaly(ecc_anom, ecc);
    } else {
        mean_anom = true_anom; // Parabolic or undefined, fallback
    }
    double mean_anom_deg = mean_anom * 180.0 / pi;
    // Mean motion (revs/day)
    double mean_motion = sqrt(mu_km / pow(sma, 3)) * 86400.0 / (2.0 * pi);
    // Robust TLE generator using correct field widths and checksums
    auto elements_to_tle = [&](double epoch_jd) -> std::pair<std::string, std::string> {
        // --- TLE fields ---
        int satnum = 99999; // Satellite catalog number
        char classification = 'U';
        int launch_year = 25; // last two digits
        int launch_num = 1;
        char launch_piece = 'A';
        int epoch_year = 25; // last two digits
        int epoch_day = 274; // day of year
        double epoch_frac = 0.12345678; // fractional day
        double mean_motion_dot = 0.0; // First derivative of mean motion
        double mean_motion_ddot = 0.0; // Second derivative of mean motion
        double bstar = 0.0; // Drag term
        int ephemeris_type = 0;
        int element_set_num = 9991;
        int rev_num = 56353; // Revolution number
        // NaN/Inf-safe orbital elements
        auto safe = [](double val, double def) {
            return (std::isnan(val) || std::isinf(val)) ? def : val;
        };
        double safe_inc_deg = safe(inc_deg, 0.0);
        double safe_raan_deg = safe(raan_deg, 0.0);
        double safe_argp_deg = safe(argp_deg, 0.0);
        double safe_mean_anom_deg = safe(mean_anom_deg, 0.0);
        double safe_mean_motion = safe(mean_motion, 1.0);
        double safe_ecc = safe(ecc, 0.0);
        // Clamp eccentricity to [0, 0.9999999]
        if (safe_ecc < 0.0) safe_ecc = 0.0;
        if (safe_ecc > 0.9999999) safe_ecc = 0.9999999;
        int ecc_int = static_cast<int>(safe_ecc * 1e7 + 0.5);
        // Clamp mean motion to reasonable minimum
        if (safe_mean_motion < 0.1) safe_mean_motion = 0.1;
        // Clamp mean anomaly to [0, 360)
        safe_mean_anom_deg = fmod(fmod(safe_mean_anom_deg, 360.0) + 360.0, 360.0);
        // --- Build Line 1 ---
        // --- Mean motion dot (10 chars, sign, leading space if positive) ---
        char mm_dot[11];
        snprintf(mm_dot, sizeof(mm_dot), "%10.8f", mean_motion_dot); // e.g. ' 0.00000000' or '-0.00000000'
        // --- Mean motion ddot (8 chars, NORAD scientific notation) ---
        char mm_ddot[9];
        int mm_ddot_sign = (mean_motion_ddot < 0) ? -1 : 1;
        double mm_ddot_abs = fabs(mean_motion_ddot);
        int mm_ddot_mant = (int)(mm_ddot_abs * 1e5 + 0.5); // 5 digits
        int mm_ddot_exp = -5; // always -5 for TLE
        snprintf(mm_ddot, sizeof(mm_ddot), "%c%05d-%1d", (mm_ddot_sign < 0 ? '-' : ' '), mm_ddot_mant, abs(mm_ddot_exp));
        // --- BSTAR (8 chars, NORAD scientific notation) ---
        char bstar_field[9];
        int bstar_sign = (bstar < 0) ? -1 : 1;
        double bstar_abs = fabs(bstar);
        int bstar_mant = (int)(bstar_abs * 1e5 + 0.5); // 5 digits
        int bstar_exp = -5; // always -5 for TLE
        snprintf(bstar_field, sizeof(bstar_field), "%c%05d-%1d", (bstar_sign < 0 ? '-' : ' '), bstar_mant, abs(bstar_exp));
        // --- Build Line 1 ---
        std::string tle1;
        tle1 += "1 "; // 1-2
        tle1 += (std::string(5 - std::to_string(satnum).length(), '0') + std::to_string(satnum)); // 3-7
        tle1 += classification; // 8
        tle1 += " "; // 9
        char intldes[8];
        snprintf(intldes, sizeof(intldes), "%02d%03d%c", launch_year, launch_num, launch_piece); // 10-16
        tle1 += intldes;
        tle1 += std::string(3, ' '); // 17-19 (spaces)
        char epoch_field[15];
        snprintf(epoch_field, sizeof(epoch_field), "%02d%03d.%08d", epoch_year, epoch_day, (int)(epoch_frac * 1e8)); // 20-32
        tle1 += epoch_field;
        tle1 += " "; // 33
        tle1 += mm_dot; // 34-43
        tle1 += " "; // 44
        tle1 += mm_ddot; // 45-52
        tle1 += " "; // 53
        tle1 += bstar_field; // 54-61
        tle1 += " "; // 62
        tle1 += std::to_string(ephemeris_type); // 63
        tle1 += " "; // 64
        tle1 += (std::string(4 - std::to_string(element_set_num).length(), '0') + std::to_string(element_set_num)); // 65-68
        // Pad to 68 chars
        while (tle1.length() < 68) tle1 += ' ';
        // --- Checksum ---
        int checksum1 = 0;
        for (char c : tle1) {
            if (isdigit(c)) checksum1 += c - '0';
            if (c == '-') checksum1 += 1;
        }
        checksum1 %= 10;
        tle1 += std::to_string(checksum1); // 69
        // --- Build Line 2 ---
        std::string tle2;
        tle2 += "2 "; // 1-2
        tle2 += (std::string(5 - std::to_string(satnum).length(), '0') + std::to_string(satnum)); // 3-7
        tle2 += " "; // 8
        char inc_field[9];
        snprintf(inc_field, sizeof(inc_field), "%8.4f", safe_inc_deg); // 9-16
        tle2 += inc_field;
        tle2 += " "; // 17
        char raan_field[9];
        snprintf(raan_field, sizeof(raan_field), "%8.4f", safe_raan_deg); // 18-25
        tle2 += raan_field;
        tle2 += " "; // 26
        char ecc_field[8];
        snprintf(ecc_field, sizeof(ecc_field), "%07d", ecc_int); // 27-33
        tle2 += ecc_field;
        tle2 += " "; // 34
        char argp_field[9];
        snprintf(argp_field, sizeof(argp_field), "%8.4f", safe_argp_deg); // 35-42
        tle2 += argp_field;
        tle2 += " "; // 43
        char mean_anom_field[9];
        snprintf(mean_anom_field, sizeof(mean_anom_field), "%8.4f", safe_mean_anom_deg); // 44-51
        tle2 += mean_anom_field;
        tle2 += " "; // 52
        char mm_field[12];
        snprintf(mm_field, sizeof(mm_field), "%11.8f", safe_mean_motion); // 53-63
        tle2 += mm_field; // mean motion (columns 53–63)
        // REMOVE the space before revolution number!
        char rev_field[6];
        int rev_num_field = rev_num % 100000;
        snprintf(rev_field, sizeof(rev_field), "%05d", rev_num_field); // 64–68
        tle2 += rev_field;
        // --- Checksum ---
        int checksum2 = 0;
        for (char c : tle2) {
            if (isdigit(c)) checksum2 += c - '0';
            if (c == '-') checksum2 += 1;
        }
        checksum2 %= 10;
        tle2 += std::to_string(checksum2); // 69
        return {tle1, tle2};
    };
    double epoch_jd = 2451545.0; // J2000, placeholder
    std::pair<std::string, std::string> tle_pair = elements_to_tle(epoch_jd);
    std::string tle_line1 = tle_pair.first;
    std::string tle_line2 = tle_pair.second;
    // Use types from libsgp4 namespace
    libsgp4::Tle tle("OBJECT_NAME", tle_line1, tle_line2);
    libsgp4::SGP4 sgp4(tle);
    libsgp4::DateTime epoch = tle.Epoch();
    libsgp4::DateTime future_dt = epoch.AddSeconds(dt);
    libsgp4::Eci eci_state = sgp4.FindPosition(future_dt);
    Eigen::VectorXd propagated_state(6);
    propagated_state(0) = eci_state.Position().x;
    propagated_state(1) = eci_state.Position().y;
    propagated_state(2) = eci_state.Position().z;
    propagated_state(3) = eci_state.Velocity().x;
    propagated_state(4) = eci_state.Velocity().y;
    propagated_state(5) = eci_state.Velocity().z;
    // Convert SGP4 output from km/km/s to meters/m/s (do NOT touch process noise logic)
    for (int i = 0; i < 3; ++i) propagated_state(i) *= 1000.0; // position: km -> m
    for (int i = 3; i < 6; ++i) propagated_state(i) *= 1000.0; // velocity: km/s -> m/s
    // Add process noise (leave this section unchanged)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    Eigen::LLT<Eigen::MatrixXd> llt(process_noise_covariance_);
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::VectorXd noise(6);
    for (int i = 0; i < 6; ++i) {
        std::normal_distribution<> d(0.0, 1.0);
        noise(i) = d(gen);
    }
    propagated_state += L * noise;
    // Write new ECI/velocity to state_vector
    for (int i = 0; i < 6; ++i) {
        propagated_particle.state_vector[i] = propagated_state[i];
    }
    return propagated_particle;
}
