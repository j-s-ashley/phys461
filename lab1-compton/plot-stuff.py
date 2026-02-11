import numpy as np
import matplotlib.pyplot as plt

e  = 662.0 # cesium-137 gamma ray [keV]
e0 = 511.0 # rest energy of electron [keV]

data = {
    10: 646.411,
    30: 381.763,
    60: 277.885,
    70: 243.942,
    100: 278.902,
    110: 250.281,
}

full_w_half_max = {
    10: 8.108,
    30: 22.29,
    60: 20,
    70: 13.314,
    100: 16.183,
    110: 13.556,
}

sketch_data = {30, 60, 70}
angles = list(data.keys())

def correct_sys_error(eprime):
    return eprime * 1.45

def degrees_to_radians(degrees):
    return degrees * (np.pi / 180.0)

def x_from_angle(deg):                 # x-axis quantity
    rads = degrees_to_radians(deg)
    return 1.0 - np.cos(rads)          # 1 - cos(theta)

def y_from_eprime(eprime):             # y-axis quantity
    return (1.0 / eprime) - (1.0 / e)  # 1/E' - 1/E

def std_dev_from_fwhm(fwhm):
    ln2 = np.log(2)
    den = 2 * np.sqrt(2 * ln2)
    return fwhm / den

def y_error(eprime, sigma):
	return sigma / (eprime*eprime)

def expected_e_from_angle(angle):
    one_less_cos_a = x_from_angle(angle)
    denominator    = 1/e0 * one_less_cos_a + 1/e
    return 1 / denominator

# ---------- Plot all data as-is ----------
x_raw = np.array([x_from_angle(a) for a in angles], dtype=float)
y_raw = np.array([y_from_eprime(data[a]) for a in angles], dtype=float)

m_raw, b_raw = np.polyfit(x_raw, y_raw, 1)

plt.figure()

raw_err = np.array([
    y_error(data[a], std_dev_from_fwhm(full_w_half_max[a]))
    for a in angles
], dtype=float)

plt.errorbar(x_raw, y_raw, yerr=raw_err, marker='.', color='blue', linestyle='none', label="Original data")
plt.plot(x_raw, m_raw * x_raw + b_raw, "g--", label=f"Fit: y = {m_raw:.4g}x + {b_raw:.4g}")
plt.xlabel(r"$1 - \cos(\theta)$")
plt.ylabel(r"$1/E' - 1/E$ [1/keV]")
plt.legend()
plt.grid(True)
plt.savefig("og_data.png", dpi=200)

# ---------- Plot corrected subset ----------
x_corr = np.array([x_from_angle(a) for a in angles], dtype=float)

# corrected y-values only for sketch angles
y_corr = np.array([
    y_from_eprime(correct_sys_error(data[a])) if a in sketch_data else y_from_eprime(data[a])
    for a in angles
], dtype=float)

m_corr, b_corr = np.polyfit(x_corr, y_corr, 1)

plt.figure()
mask = np.array([a in sketch_data for a in angles])

cor_err = np.array([
    correct_sys_error(y_error(data[a], std_dev_from_fwhm(full_w_half_max[a])))
    for a in angles
], dtype=float)

plt.errorbar(x_corr[~mask], y_corr[~mask], yerr=raw_err[~mask], marker='.', color='blue', linestyle='none', label="Uncorrected points")
plt.errorbar(x_corr[mask],  y_corr[mask], yerr=cor_err[mask], marker='.', color='orange', linestyle='none', label="Corrected points")
plt.plot(x_corr, m_corr * x_corr + b_corr, "g--", label=f"Fit: y = {m_corr:.4g}x + {b_corr:.4g}")

plt.xlabel(r"$1 - \cos(\theta)$")
plt.ylabel(r"$1/E' - 1/E$ [1/keV]")
plt.legend()
plt.grid(True)
plt.savefig("cr_data.png", dpi=200)

# ---------- Compare expected and measured energies ----------
x_exp = np.array([a for a in angles], dtype=float)
y_exp = np.array([expected_e_from_angle(a) for a in angles], dtype=float)
y_act = np.array([data[a] for a in angles], dtype=float)

plt.figure()

act_err = np.array([std_dev_from_fwhm(full_w_half_max[a]) for a in angles], dtype=float) # get error
plt.errorbar(x_exp, y_act, yerr=act_err, marker='.', color='green', linestyle='none', label="Actual scattered energies")
plt.scatter(x_exp, y_exp, marker='.', color='blue', label="Expected scattered energies")
plt.xlabel(r"Angle [$\degree$]")
plt.ylabel("Energy [keV]")
plt.legend()
plt.grid(True)
plt.savefig("angle_vs_expected_energies.png", dpi=200)

# ---------- Compare expected and corrected measured energies ----------
x_exp = np.array([a for a in angles], dtype=float)
y_exp = np.array([expected_e_from_angle(a) for a in angles], dtype=float)
y_act = np.array([data[a] for a in angles], dtype=float)

# corrected y-values only for sketch angles
y_cor = np.array([
    correct_sys_error(data[a]) if a in sketch_data else data[a]
    for a in angles
], dtype=float)

mask = np.array([a in sketch_data for a in angles])

cor_e_err = np.array([
    correct_sys_error(std_dev_from_fwhm(full_w_half_max[a]))
    for a in angles
], dtype=float)

plt.figure()

act_err = np.array([std_dev_from_fwhm(full_w_half_max[a]) for a in angles], dtype=float) # get error
plt.errorbar(x_exp[~mask], y_act[~mask], yerr=act_err[~mask], marker='.', color='green', linestyle='none', label="Actual scattered energies")
plt.errorbar(x_exp[mask],  y_cor[mask], yerr=cor_e_err[mask], marker='.', color='orange', linestyle='none', label="Corrected scattered energies")

plt.scatter(x_exp, y_exp, marker='.', color='blue', label="Expected scattered energies")
plt.xlabel(r"Angle [$\degree$]")
plt.ylabel("Energy [keV]")
plt.legend()
plt.grid(True)
plt.savefig("angle_vs_expected_energies_cor.png", dpi=200)
