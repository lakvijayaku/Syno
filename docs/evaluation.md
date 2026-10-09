## Evaluation

This document defines how SYNO's behavior is measured. "Human-like" is not directly testable, so success is defined as reproducing behavioral signatures documented in biological research. Each signature must emerge without being hard-coded.

### Behavioral Signatures
| ID | Signature | Expected Result | Source | Status |
|---|---|---|---|---|
| **E1** | Satiation | Food intake slows gradually during a meal rather than stopping abruptly. | [`biology.md`](biology.md), Homeostatic Regulation | Reproduced, weakly (7 of 10 seeds): a half-hungry SYNO standing on food prefers to eat less as its stomach fills, from 0.98 when empty to 0.62 when full on average, rather than stopping abruptly. Seeds differ widely: when full, one seed never eats and another always does (`experiments/signatures.py`) |
| **E2** | State-Dependent Eating | The probability of eating rises with energy deficit. | [`biology.md`](biology.md), State-Dependent Reward | Not reproduced (2 of 10 seeds): a trained SYNO standing on food prefers to eat in 0.99 of cases when starving and still 0.89 when full on average. Only 2 seeds eat clearly less when full (`experiments/signatures.py`) |
| **E3** | Partial-Fullness Eating | A partially full SYNO eats nearby food but ignores distant food. | [`biology.md`](biology.md), State-Dependent Reward | Not reproduced (1 of 10 seeds): at energy 0.8, SYNO moves toward visible food 3 or more steps away nearly as often as toward adjacent food (0.91 against 0.93 on average) (`experiments/signatures.py`) |
| **E4** | RPE Transfer | With training, the DS response shifts from eating to seeing food. | [`biology.md`](biology.md), Reward Prediction Error | Reproduced (10 of 10 seeds): the RPE when eating falls from 2.335 before training to −0.027 after 3000 lives, while the RPE when food comes into view rises from −0.278 to +0.183, and is positive in every seed (`experiments/signatures.py`) |
| **E5** | Reward Omission | When expected food is missing, the DS produces a negative RPE. | [`biology.md`](biology.md), Reward Prediction Error | Reproduced (10 of 10 seeds): when food SYNO is standing on vanishes just before it eats, its RPE is −1.337 on average, against −0.027 when the meal arrives, and dips by more than 1.1 in every seed (`experiments/signatures.py`) |
| **E6** | Habituation | Exploration of a familiar area declines over time and recovers after absence. | [`biology.md`](biology.md), Novelty and Habituation | Not measurable with the current design: the Novelty System habituates and recovers, but familiarity is not an input to the DN, so SYNO cannot act on it, and SYNO's 5x5 world is crossed within a few steps, so there is no unfamiliar area to return to |

### Score
These results feed the capability scores in [`humanness.md`](humanness.md), which covers every capability of a human mind and body, not only the measured signatures.

### Method
Signatures are measured with recorded data from experiment runs. A signature passes only when the result is reproducible across multiple runs with different random seeds. Signatures are measured in `experiments/signatures.py`, which trains SYNO in the scarce world (`experiments/scarcity.py`) with 10 seeds, in parallel, and probes each trained network in hand-made situations, counting only its preferred action. Run it with `python3 -m experiments.signatures`.

Each result is reported as its average and its range over the seeds, followed by how many seeds show the signature on their own. A seed shows a rate-based signature when the rate changes by at least 0.1, and E5 when the RPE dips by at least 0.5. A signature is **reproduced** when 8 or more of the 10 seeds show it, **reproduced weakly** when 4 to 7 do, and **not reproduced** when 3 or fewer do.
