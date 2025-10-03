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

Particle SGP4Propagator::propagate(const Particle& particle, double dt, double current_time) const {
    Particle propagated_particle = particle;
    // Extract ECI position and velocity from state_vector
    Eigen::Vector3d pos_m = particle.state_vector.head(3); // meters
    Eigen::Vector3d vel_ms = particle.state_vector.segment(3, 3); // m/s
    double bstar = particle.state_vector(6); // Ballistic coefficient
    // Constants
    const double pi = 3.14159265358979323846;
    const double mu_km = 398600.4418; // km^3/s^2
    // Convert to km and km/s for TLE/SGP4
    Eigen::Vector3d pos_km = pos_m / 1000.0;
    Eigen::Vector3d vel_kms = vel_ms / 1000.0;
    Eigen::Matrix<double, 6, 1> cartesian_km;
    cartesian_km << pos_km, vel_kms;
    auto kep = astro::convertCartesianToKeplerianElements(cartesian_km, mu_km);
    double sma = kep[astro::semiMajorAxisIndex];
    double ecc = kep[astro::eccentricityIndex];
    // SGP4 is mathematically undefined for non-elliptical (parabolic or hyperbolic) orbits.
    // Process noise can occasionally push a particle's state into this regime (ecc >= 1.0 or sma <= 0).
    // Attempting to generate a TLE from such a state will produce NaNs and cause catastrophic
    // propagator divergence. The robust solution is to detect this case and fall back to a
    // simple, stable linear propagation for that particle for that single time step
    if (ecc >= 1.0 || sma <= 0.0) {
        // Fallback: Use linear propagation (x_new = x + v*dt)
        Eigen::VectorXd new_state = particle.state_vector;
        new_state.head(3) += new_state.segment(3, 3) * dt;
        
        // Add the original process noise to the linearly propagated state
        static std::random_device rd;
        static std::mt19937 gen(rd());
        Eigen::LLT<Eigen::MatrixXd> llt(process_noise_covariance_);
        Eigen::MatrixXd L = llt.matrixL();
        Eigen::VectorXd noise(7);
        for (int i = 0; i < 7; ++i) {
            std::normal_distribution<> d(0.0, 1.0);
            noise(i) = d(gen);
        }
        new_state += L * noise;
        
        propagated_particle.state_vector = new_state;
        return propagated_particle; // Return early, skipping the SGP4 logic
    }
    double inc_deg = kep[astro::inclinationIndex] * 180.0 / pi;
    double raan_deg = kep[astro::longitudeOfAscendingNodeIndex] * 180.0 / pi;
    double argp_deg = kep[astro::argumentOfPeriapsisIndex] * 180.0 / pi;
    double true_anom = kep[astro::trueAnomalyIndex];
    // Handle near-equatorial and near-circular orbit singularities which can
    // cause the astro library to return garbage values for RAAN and Arg of Perigee.
    const double singularity_threshold = 1e-6;

    if (inc_deg < singularity_threshold) {
        // For near-equatorial orbits, RAAN is ill-defined. Set to 0 by convention.
        raan_deg = 0.0;
        if (ecc < singularity_threshold) {
            // For near-circular equatorial orbits, Arg of Perigee is also ill-defined.
            // The anomaly can be measured from the X-axis (longitude of perigee = 0).
            argp_deg = 0.0;
        }
    } else {
        if (ecc < singularity_threshold) {
            // For near-circular inclined orbits, Arg of Perigee is ill-defined.
            // Set to 0 by convention, anomaly measured from the ascending node.
            argp_deg = 0.0;
        }
    }
    double mean_anom = 0.0;
    if (ecc < 1.0) {
        double ecc_anom = astro::convertTrueAnomalyToEllipticalEccentricAnomaly(true_anom, ecc);
        mean_anom = astro::convertEllipticalEccentricAnomalyToMeanAnomaly(ecc_anom, ecc);
    } else if (ecc > 1.0) {
        double ecc_anom = astro::convertTrueAnomalyToHyperbolicEccentricAnomaly(true_anom, ecc);
        mean_anom = astro::convertHyperbolicEccentricAnomalyToMeanAnomaly(ecc_anom, ecc);
    } else {
        mean_anom = true_anom;
    }
    double mean_anom_deg = mean_anom * 180.0 / pi;
    double mean_motion = sqrt(mu_km / pow(sma, 3)) * 86400.0 / (2.0 * pi);
    // --- Calculate epoch from current_time ---
    // The Julian date for the Unix epoch (1970-01-01 00:00:00 UTC) is 2440587.5
    double epoch_jd = 2440587.5 + current_time / 86400.0;
    // Convert Julian Date to TLE epoch format
    int jd_int = static_cast<int>(epoch_jd + 0.5);
    double jd_frac = epoch_jd + 0.5 - jd_int;
    int l = jd_int + 68569;
    int n = 4 * l / 146097;
    l = l - (146097 * n + 3) / 4;
    int i = 4000 * (l + 1) / 1461001;
    l = l - 1461 * i / 4 + 31;
    int j = 80 * l / 2447;
    int day = l - 2447 * j / 80;
    l = j / 11;
    int month = j + 2 - 12 * l;
    int year = 100 * (n - 49) + i + l;
    // Calculate day of year (epoch_day)
    int days_in_month[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
        days_in_month[2] = 29; // Leap year
    }
    int epoch_day = 0;
    for (int m = 1; m < month; ++m) {
        epoch_day += days_in_month[m];
    }
    epoch_day += day;
    int epoch_year = year % 100; // Last two digits of the year
    double epoch_frac = jd_frac;
    auto elements_to_tle = [&](double epoch_jd) -> std::pair<std::string, std::string> {
        // --- TLE fields ---
        int satnum = 99999; // Satellite catalog number
        char classification = 'U';
        int launch_year = epoch_year; // last two digits
        int launch_num = 1;
        char launch_piece = 'A';
        // Use calculated epoch_year, epoch_day, epoch_frac
        double mean_motion_dot = 0.0;
        double mean_motion_ddot = 0.0;
        double bstar = particle.state_vector(6);
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
        safe_raan_deg = fmod(fmod(safe_raan_deg, 360.0) + 360.0, 360.0);
        safe_argp_deg = fmod(fmod(safe_argp_deg, 360.0) + 360.0, 360.0);
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
        double mean_motion_ddot_val = mean_motion_ddot; // Use a temp variable
        if (std::abs(mean_motion_ddot_val) < 1e-9) {
            snprintf(mm_ddot, sizeof(mm_ddot), " 00000+0"); // Use a standard zero representation
        } else {
            double mm_ddot_abs = std::abs(mean_motion_ddot_val);
            int mm_ddot_exp = static_cast<int>(floor(log10(mm_ddot_abs)));
            int mm_ddot_mant = static_cast<int>(round(mm_ddot_abs / pow(10, mm_ddot_exp - 5)));

            // --- CRITICAL NORMALIZATION STEP ---
            while (mm_ddot_mant >= 100000 && mm_ddot_exp < 9) {
                mm_ddot_mant = static_cast<int>(round(static_cast<double>(mm_ddot_mant) / 10.0));
                mm_ddot_exp += 1;
            }
            
            // Clamp exponent to valid TLE range [-9, 9]
            if (mm_ddot_exp > 9) mm_ddot_exp = 9;
            if (mm_ddot_exp < -9) mm_ddot_exp = -9;
            
            char mm_ddot_sign = (mean_motion_ddot_val < 0) ? '-' : ' ';
            char mm_ddot_exp_sign = (mm_ddot_exp < 0) ? '-' : '+';
            snprintf(mm_ddot, sizeof(mm_ddot), "%c%05d%c%1d", mm_ddot_sign, mm_ddot_mant, mm_ddot_exp_sign, abs(mm_ddot_exp));
        }
        // --- BSTAR (8 chars, NORAD scientific notation) ---
        char bstar_field[9];
        double bstar_val = bstar; // Use a temp variable
        if (std::abs(bstar_val) < 1e-9) {
            snprintf(bstar_field, sizeof(bstar_field), " 00000+0"); // Use a standard zero representation
        } else {
            double bstar_abs = std::abs(bstar_val);
            int bstar_exp = static_cast<int>(floor(log10(bstar_abs)));
            int bstar_mant = static_cast<int>(round(bstar_abs / pow(10, bstar_exp - 5)));

            // --- CRITICAL NORMALIZATION STEP ---
            while (bstar_mant >= 100000 && bstar_exp < 9) {
                bstar_mant = static_cast<int>(round(static_cast<double>(bstar_mant) / 10.0));
                bstar_exp += 1;
            }

            // Clamp exponent to valid TLE range [-9, 9]
            if (bstar_exp > 9) bstar_exp = 9;
            if (bstar_exp < -9) bstar_exp = -9;
            
            char bstar_sign = (bstar_val < 0) ? '-' : ' ';
            char bstar_exp_sign = (bstar_exp < 0) ? '-' : '+';
            snprintf(bstar_field, sizeof(bstar_field), "%c%05d%c%1d", bstar_sign, bstar_mant, bstar_exp_sign, abs(bstar_exp));
        }
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
    std::pair<std::string, std::string> tle_pair = elements_to_tle(epoch_jd);
    std::string tle_line1 = tle_pair.first;
    std::string tle_line2 = tle_pair.second;
    libsgp4::Tle tle("OBJECT_NAME", tle_line1, tle_line2);
    libsgp4::SGP4 sgp4(tle);
    libsgp4::DateTime epoch = tle.Epoch();
    libsgp4::DateTime future_dt = epoch.AddSeconds(dt);
    libsgp4::Eci eci_state = sgp4.FindPosition(future_dt);
    Eigen::VectorXd propagated_state(7);
    propagated_state(0) = eci_state.Position().x * 1000.0;
    propagated_state(1) = eci_state.Position().y * 1000.0;
    propagated_state(2) = eci_state.Position().z * 1000.0;
    propagated_state(3) = eci_state.Velocity().x * 1000.0;
    propagated_state(4) = eci_state.Velocity().y * 1000.0;
    propagated_state(5) = eci_state.Velocity().z * 1000.0;
    propagated_state(6) = particle.state_vector(6); // Propagate bc unchanged
    // Add process noise (7D)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    Eigen::LLT<Eigen::MatrixXd> llt(process_noise_covariance_);
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::VectorXd noise(7);
    for (int i = 0; i < 7; ++i) {
        std::normal_distribution<> d(0.0, 1.0);
        noise(i) = d(gen);
    }
    propagated_state += L * noise;
    propagated_particle.state_vector = propagated_state;
    return propagated_particle;
}
