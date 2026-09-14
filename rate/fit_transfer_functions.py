"""
Derive the rate-model transfer functions F(I) = k * [I - theta]_+^n for each
cortical cell type, by measuring the steady-state f-I curve of the
corresponding Izhikevich cell and fitting a threshold power law.

IMPORTANT: this uses the Izhikevich-2007 form
    C dv/dt = k(v - v_r)(v - v_t) - u + I
      du/dt = a( b(v - v_r) - u )
    reset when v >= v_peak:  v <- c,  u <- u + d
which is the form the network's cell parameters (C, k, v_r, v_t and the
a/b/c/d ranges) actually belong to. The repo's spiking model/Neuron.py instead
integrates the older Izhikevich-2003 form (0.04 v^2 + 5 v + 140 - u + I) with
these 2007 parameters, a mismatch that makes FS/LTS cells fire spontaneously at
rest and inflates rates into the thousands (see docs in the rate/ package).
We do NOT modify Neuron.py (the spiking model is preserved as the "Last Spiking"
checkpoint); this standalone measurement only produces the fitted transfer
coefficients baked into CellTypes.py.

Run directly to re-derive and print the coefficients:
    python3 rate/fit_transfer_functions.py
"""
import numpy as np
from scipy.optimize import curve_fit

# Izhikevich-2007 parameters per cortical cell type, matching the values used
# in network/RetinotopicAVNetwork.py and network/SerotoninAVNetwork.py.
CELL_PARAMS = {
    "Pyramidal": {"C": 100, "k": 3, "v_r": -60, "v_t": -50, "v_peak": 50,
                  "a": 0.01, "b": 5, "c": -60, "d": 400},
    "FS": {"C": 20, "k": 1, "v_r": -55, "v_t": -40, "v_peak": 25,
           "a": 0.15, "b": 8, "c": -55, "d": 200},
    "LTS": {"C": 100, "k": 1, "v_r": -56, "v_t": -42, "v_peak": 40,
            "a": 0.03, "b": 8, "c": -50, "d": 200},
}

TAU = 0.1            # ms, matches the network integration step
T_TOTAL = 2000.0     # ms per current level
T_DISCARD = 1000.0   # ms of leading transient discarded before counting spikes
# Rates at/above this are contaminated by fixed-timestep inter-spike-interval
# quantization (spike-every-few-steps staircase); excluded from the fit.
QUANT_CEILING = 2000.0


def _derivs(v, u, p, I):
    dv = (p["k"] * (v - p["v_r"]) * (v - p["v_t"]) - u + I) / p["C"]
    du = p["a"] * (p["b"] * (v - p["v_r"]) - u)
    return dv, du


def measure_rate(p, I):
    """Steady-state firing rate (spikes/sec) of one Izhikevich-2007 cell at constant I."""
    v, u, t = p["v_r"], 0.0, 0.0
    spikes = 0
    h = TAU
    for _ in range(int(T_TOTAL / TAU)):
        t += h
        k1v, k1u = _derivs(v, u, p, I)
        k2v, k2u = _derivs(v + 0.5 * h * k1v, u + 0.5 * h * k1u, p, I)
        k3v, k3u = _derivs(v + 0.5 * h * k2v, u + 0.5 * h * k2u, p, I)
        k4v, k4u = _derivs(v + h * k3v, u + h * k3u, p, I)
        v += (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        u += (h / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        if v >= p["v_peak"]:
            v, u = p["c"], u + p["d"]
            if t >= T_DISCARD:
                spikes += 1
    return spikes / ((T_TOTAL - T_DISCARD) / 1000.0)


def power_law(I, k, theta, n):
    return k * np.clip(I - theta, 0, None) ** n


def fit_cell_type(name, currents=None):
    if currents is None:
        currents = np.linspace(0, 1400, 57)
    currents = np.asarray(currents, dtype=float)
    rates = np.array([measure_rate(CELL_PARAMS[name], I) for I in currents])

    clean = rates < QUANT_CEILING
    fit_I, fit_r = currents[clean], rates[clean]
    positive = fit_r > 0
    theta0 = fit_I[positive][0]
    span = max(fit_I[positive][-1] - theta0, 1.0)
    p0 = [fit_r.max() / (span ** 0.5), theta0, 0.5]
    popt, _ = curve_fit(power_law, fit_I, fit_r, p0=p0, maxfev=40000,
                        bounds=([0, -np.inf, 0.2], [np.inf, np.inf, 4.0]))
    resid = fit_r - power_law(fit_I, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((fit_r - fit_r.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"k": float(popt[0]), "theta": float(popt[1]), "n": float(popt[2]),
            "r2": r2, "currents": currents, "rates": rates}


if __name__ == "__main__":
    for name in CELL_PARAMS:
        fit = fit_cell_type(name)
        print("%-10s k=%.5g  theta=%.5g  n=%.5g  R^2=%.4f"
              % (name, fit["k"], fit["theta"], fit["n"], fit["r2"]))
