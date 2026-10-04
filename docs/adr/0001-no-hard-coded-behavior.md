## ADR 0001: No Hard-Coded Behavior

### Status
Accepted

### Context
SYNO's goal is to produce behavior similar to that of a living organism, such as eating, exploring, and resting. These behaviors could be scripted directly (for example, "eat when hunger exceeds 50%"), but scripted behavior demonstrates nothing about how motivation arises and cannot adapt to new situations.

### Decision
Only physiology is hard-coded: internal variables, their dynamics, and the signals released in response. Reward is defined as a reduction in homeostatic deficit [`biology.md`](../biology.md). All behavior must emerge from learning driven by the RPE.

### Consequences
- SYNO's behavior is a genuine test of the model, since it cannot be produced by a script.
- Training is required before any meaningful behavior appears.
- Incorrect behavior must be fixed by adjusting physiology or learning, never by adding behavioral rules.
- Success must be measured against the signatures in [`evaluation.md`](../evaluation.md).
