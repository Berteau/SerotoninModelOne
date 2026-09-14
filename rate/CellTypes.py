"""
Single source of truth for the rate model's per-cell-type transfer functions
and membrane time constants.

The transfer coefficients (k, theta, n) below were fit by
rate/fit_transfer_functions.py to the steady-state f-I curves of the
Izhikevich-2007 cells (see that file for the important note about the 2003/2007
integration mismatch in the spiking model). Re-run that script to regenerate
them; the values here should match its output.

Transfer function (steady-state firing rate as a function of total input
current I, in spikes/sec):

    F(I) = k * max(I - theta, 0) ** n

Membrane time constants tau_m set how fast a cell's rate relaxes toward F(I):

    tau_m * dr/dt = -r + F(I)

They are chosen from typical cortical values (FS cells fastest), not fit, and
are exposed here so they can be tuned in one place.
"""

CELL_TYPES = {
    "Pyramidal": {"k": 0.082284, "theta": 59.51, "n": 0.82971, "tau_m": 20.0},
    "FS":        {"k": 0.61624,  "theta": 57.604, "n": 0.94659, "tau_m": 10.0},
    "LTS":       {"k": 0.083716, "theta": 42.616, "n": 1.0089,  "tau_m": 15.0},
}


def transfer(cell_type, I):
    """Steady-state rate F(I) = k*[I-theta]_+^n for the named cell type."""
    p = CELL_TYPES[cell_type]
    x = I - p["theta"]
    if x <= 0:
        return 0.0
    return p["k"] * (x ** p["n"])
