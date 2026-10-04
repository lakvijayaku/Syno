## Evaluation

This document defines how SYNO's behavior is measured. "Human-like" is not directly testable, so success is defined as reproducing behavioral signatures documented in biological research. Each signature must emerge without being hard-coded.

### Behavioral Signatures
| ID | Signature | Expected Result | Source |
|---|---|---|---|
| **E1** | Satiation | Food intake slows gradually during a meal rather than stopping abruptly. | [`biology.md`](biology.md), Homeostatic Regulation |
| **E2** | State-Dependent Eating | The probability of eating rises with energy deficit. | [`biology.md`](biology.md), State-Dependent Reward |
| **E3** | Partial-Fullness Eating | A partially full SYNO eats nearby food but ignores distant food. | [`biology.md`](biology.md), State-Dependent Reward |
| **E4** | RPE Transfer | With training, the DS response shifts from eating to seeing food. | [`biology.md`](biology.md), Reward Prediction Error |
| **E5** | Reward Omission | When expected food is missing, the DS produces a negative RPE. | [`biology.md`](biology.md), Reward Prediction Error |
| **E6** | Habituation | Exploration of a familiar area declines over time and recovers after absence. | [`biology.md`](biology.md), Novelty and Habituation |

### Method
Each signature will be measured with logged data and plots (roadmap step 6). A signature passes only when the result is reproducible across multiple runs with different random seeds.
