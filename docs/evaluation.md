## Evaluation

This document defines how SYNO's behavior is measured. "Human-like" is not directly testable, so success is defined as reproducing behavioral signatures documented in biological research. Each signature must emerge without being hard-coded.

### Behavioral Signatures
| ID | Signature | Expected Result | Source | Status |
|---|---|---|---|---|
| **E1** | Satiation | Food intake slows gradually during a meal rather than stopping abruptly. | [`biology.md`](biology.md), Homeostatic Regulation | Not measured |
| **E2** | State-Dependent Eating | The probability of eating rises with energy deficit. | [`biology.md`](biology.md), State-Dependent Reward | Not reproduced: across 5 seeds, a trained SYNO standing on food prefers to eat at every energy from 0.0 to 0.9, and only at full energy does the rate fall (to 0.73). The rate does not rise gradually with deficit (`experiments/signatures.py`) |
| **E3** | Partial-Fullness Eating | A partially full SYNO eats nearby food but ignores distant food. | [`biology.md`](biology.md), State-Dependent Reward | Not measured |
| **E4** | RPE Transfer | With training, the DS response shifts from eating to seeing food. | [`biology.md`](biology.md), Reward Prediction Error | Partly reproduced: across 5 seeds, the RPE when eating falls from 2.443 before training to −0.083 after 1500 lives, but the RPE when food appears stays negative (−0.193 to −0.063). On the 3x3 grid, food is always in view and reappears as soon as it is eaten, so its appearance never predicts anything new (`experiments/signatures.py`) |
| **E5** | Reward Omission | When expected food is missing, the DS produces a negative RPE. | [`biology.md`](biology.md), Reward Prediction Error | Reproduced: across 5 seeds, when food SYNO is standing on vanishes just before it eats, its RPE is −0.874, against −0.083 when the meal arrives (`experiments/signatures.py`) |
| **E6** | Habituation | Exploration of a familiar area declines over time and recovers after absence. | [`biology.md`](biology.md), Novelty and Habituation | Not measurable with the current design: the Novelty System habituates and recovers, but familiarity is not an input to the DN, so SYNO cannot act on it, and on the 3x3 grid every square is visited within a few steps, so there is no unfamiliar area to return to |

### Method
Signatures are measured with recorded data from experiment runs. A signature passes only when the result is reproducible across multiple runs with different random seeds. Signatures are measured in `experiments/signatures.py`, which trains SYNO as in the consolidation experiment with several seeds and probes the trained network in hand-made situations, counting only its preferred action. Run it with `python3 -m experiments.signatures`.
