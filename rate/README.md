# Rate-based reformulation (`rate/`)

A parallel, per-neuron **firing-rate** reformulation of the spiking serotonin
model, built to remove the runaway / timestep-quantization behavior of the
spiking version while staying faithful to the methods draft's mechanisms.
The spiking model under `model/` is untouched and preserved as the
"Last Spiking" checkpoint.

## Core pieces

- **`CellTypes.py`** — per-cell-type transfer functions `F(I) = k·[I−θ]₊^n`
  and membrane time constants. Coefficients are fit by
  `fit_transfer_functions.py`.
- **`fit_transfer_functions.py`** — derives the transfer coefficients from the
  steady-state f-I curves of the Izhikevich cells, using the **Izhikevich-2007**
  form the cell parameters actually belong to. (The spiking `model/Neuron.py`
  integrates the older 2003 form with 2007 parameters — a mismatch that makes
  FS/LTS fire spontaneously at rest and inflates rates into the thousands.
  Flagged for the paper authors; not "fixed" here since we don't touch the
  spiking model.)
- **`RateNeuron.py`** — `RateNeuron` (`τ_m·dr/dt = −r + F(I)`) and
  `RateInputNeuron` (directly-settable input rate).
- **`RateAxon.py`** — rate-driven saturating synaptic conductance (AMPA/NMDA/
  GABA), with **transmission failure rolled into a deterministic effective
  weight** `w_eff = w·clamp(1−failureRate, 0, 1)`, plus the calcium/NMDA
  plasticity rule.
- **`RateDiffuseReceptor.py`** — somatic (adds current) and axonal (scales
  `w_eff` via failure rate) serotonin receptors.
- **`RatePopulation.py`** — per-neuron population with connection management,
  mean-rate and cross-modal-influence recording, two-phase stepping.

## Key properties vs. the spiking model

- **Deterministic** — no stochastic spike failures, so a single run is
  representative (the spiking model needed 10× runs + confidence intervals to
  average out that noise).
- **Bounded** — saturating conductances and a sane transfer function replace
  the unbounded / quantization-driven spiking dynamics.

## Not yet done

Network/simulation assembly and parameter retuning (the spiking weights were
tuned against inflated rates and do not carry over directly).
