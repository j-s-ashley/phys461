import numpy as np
import matplotlib.pyplot as plt

e = 662.0  # keV

data = {
    10: 646.411,
    30: 381.763,
    60: 277.885,
    70: 243.942,
    100: 278.902,
    110: 250.281,
}
sketch_data = {30, 60, 70}
angles = list(data.keys())

def correct_sys_error(eprime):
    return eprime * 1.4

def degrees_to_radians(degrees):
    return degrees * (np.pi / 180.0)

def x_from_angle(deg):                 # x-axis quantity
    rads = degrees_to_radians(deg)
    return 1.0 - np.cos(rads)          # 1 - cos(theta)

def y_from_eprime(eprime):             # y-axis quantity
    return (1.0 / eprime) - (1.0 / e)  # 1/E' - 1/E

# ---------- Plot all data as-is ----------
x_raw = np.array([x_from_angle(a) for a in angles], dtype=float)
y_raw = np.array([y_from_eprime(data[a]) for a in angles], dtype=float)

m_raw, b_raw = np.polyfit(x_raw, y_raw, 1)

plt.figure()
plt.scatter(x_raw, y_raw, label="Original data")
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

plt.scatter(x_corr[~mask], y_corr[~mask], label="Uncorrected points")
plt.scatter(x_corr[mask],  y_corr[mask],  label="Corrected points")
plt.plot(x_corr, m_corr * x_corr + b_corr, "g--", label=f"Fit: y = {m_corr:.4g}x + {b_corr:.4g}")

plt.xlabel(r"$1 - \cos(\theta)$")
plt.ylabel(r"$1/E' - 1/E$ [1/keV]")
plt.legend()
plt.grid(True)
plt.savefig("cr_data.png", dpi=200)
