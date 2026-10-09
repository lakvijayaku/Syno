## Evaluation

This document defines how SYNO's behavior is measured. "Human-like" is not directly testable, so success is defined as reproducing behavioral signatures documented in biological research. Each signature must emerge without being hard-coded.

### Behavioral Signatures
| ID | Signature | Expected Result | Source | Status |
|---|---|---|---|---|
| **E1** | Satiation | Food intake slows gradually during a meal rather than stopping abruptly. | [`biology.md`](biology.md), Homeostatic Regulation | Reproduced, weakly: across 5 seeds, a half-hungry SYNO standing on food prefers to eat less often as its stomach fills (0.98 when empty, then 0.94, 0.92, 0.86, and 0.70 when full), rather than stopping abruptly. The effect is small until the stomach is nearly full (`experiments/signatures.py`) |
| **E2** | State-Dependent Eating | The probability of eating rises with energy deficit. | [`biology.md`](biology.md), State-Dependent Reward | Not reproduced: across 5 seeds, a trained SYNO standing on food prefers to eat in 0.98 to 0.99 of cases from energy 0.0 to 0.5, falling only to 0.91 when full. The rate barely changes with deficit (`experiments/signatures.py`) |
| **E3** | Partial-Fullness Eating | A partially full SYNO eats nearby food but ignores distant food. | [`biology.md`](biology.md), State-Dependent Reward | Not reproduced: across 5 seeds, SYNO moves toward visible food 3 or more steps away slightly less often than toward adjacent food (0.92 against 0.96 when hungry), but the gap does not grow as it fills (0.90 against 0.92 at energy 0.8) (`experiments/signatures.py`) |
| **E4** | RPE Transfer | With training, the DS response shifts from eating to seeing food. | [`biology.md`](biology.md), Reward Prediction Error | Reproduced: across 5 seeds, the RPE when eating falls from 2.415 before training to −0.022 after 3000 lives, while the RPE when food comes into view rises from −0.173 to +0.174, so the response moves from eating to seeing food (`experiments/signatures.py`) |
| **E5** | Reward Omission | When expected food is missing, the DS produces a negative RPE. | [`biology.md`](biology.md), Reward Prediction Error | Reproduced: across 5 seeds, when food SYNO is standing on vanishes just before it eats, its RPE is −1.305, against −0.022 when the meal arrives (`experiments/signatures.py`) |
| **E6** | Habituation | Exploration of a familiar area declines over time and recovers after absence. | [`biology.md`](biology.md), Novelty and Habituation | Not measurable with the current design: the Novelty System habituates and recovers, but familiarity is not an input to the DN, so SYNO cannot act on it, and SYNO's 5x5 world is crossed within a few steps, so there is no unfamiliar area to return to |

### Score
The results are combined into a single score in [`humanness.md`](humanness.md).

### Method
Signatures are measured with recorded data from experiment runs. A signature passes only when the result is reproducible across multiple runs with different random seeds. Signatures are measured in `experiments/signatures.py`, which trains SYNO in the scarce world (`experiments/scarcity.py`) with several seeds and probes the trained network in hand-made situations, counting only its preferred action. Run it with `python3 -m experiments.signatures`.
