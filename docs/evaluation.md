## Evaluation

This document defines how SYNO's behavior is measured. "Human-like" is not directly testable, so success is defined as reproducing behavioral signatures documented in biological research. Each signature must emerge without being hard-coded.

### Behavioral Signatures
| ID | Signature | Expected Result | Source | Status |
|---|---|---|---|---|
| **E1** | Satiation | Food intake slows gradually during a meal rather than stopping abruptly. | [`biology.md`](biology.md), Homeostatic Regulation | Not measured |
| **E2** | State-Dependent Eating | The probability of eating rises with energy deficit. | [`biology.md`](biology.md), State-Dependent Reward | Preliminary: a trained SYNO eats whenever energy is 0.7 or below, and sometimes declines food near full energy |
| **E3** | Partial-Fullness Eating | A partially full SYNO eats nearby food but ignores distant food. | [`biology.md`](biology.md), State-Dependent Reward | Not measured |
| **E4** | RPE Transfer | With training, the DS response shifts from eating to seeing food. | [`biology.md`](biology.md), Reward Prediction Error | Verified for the RPE formula only, not yet in behavior |
| **E5** | Reward Omission | When expected food is missing, the DS produces a negative RPE. | [`biology.md`](biology.md), Reward Prediction Error | Verified for the RPE formula only, not yet in behavior |
| **E6** | Habituation | Exploration of a familiar area declines over time and recovers after absence. | [`biology.md`](biology.md), Novelty and Habituation | Verified for the Novelty System only, not yet in behavior |

### Method
Signatures are measured with recorded data from experiment runs. A signature passes only when the result is reproducible across multiple runs with different random seeds. The recording tools are in progress (roadmap step 6), so no signature has been formally measured yet.
