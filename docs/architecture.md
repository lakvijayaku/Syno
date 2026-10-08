## Architecture

This document describes how SYNO is structured and how its components interact. Component names are defined in [`glossary.md`](glossary.md). The biological basis for each component is described in [`biology.md`](biology.md).

### Design Principle
SYNO's behavior is never hard-coded. Only the physiology is defined: internal body variables, how they change over time, and the signals released in response. All behavior, including when to eat, explore, or rest, must emerge from learning. See [`ADR 0001`](adr/0001-no-hard-coded-behavior.md).

### Overview
SYNO consists of the **Habitat (HB)**, which is the external world, a body, the **Homeostatic Core (HC)**, and the **Artificial Mind (AM)**, which perceives, decides, and learns. The code is organized the same way:

| Package | Contains |
|---|---|
| `syno/world/` | The Habitat: the grid, movement, eating, and senses |
| `syno/body/` | The Homeostatic Core: energy, stomach, drive, and reward |
| `syno/brain/` | Neurons, layers, networks, training, the Dopamine System, action selection, the Novelty System, the Memory Store, and the Growth Engine |
| `syno/tools/` | Recording data from runs, smoothing it, and drawing it as charts |

```
┌──────────────────────── HB ────────────────────────┐
│  food and open space, bounded by walls             │
└──────────┬──────────────────────────────▲──────────┘
     senses│                              │action
┌──────────▼─────────────── AM ───────────┴──────────┐
│          SI ─────────────► DN ──────────┘          │
│                            ▲ ▲                     │
│   HC ──────────────────────┘ │ RPE                 │
│   │                          │                     │
│   └──► reward ──► DS ────────┘                     │
│          ▲        ▲                                │
│          NS       └── DN's own value prediction    │
└────────────────────────────────────────────────────┘
```

The **Value Estimator (VE)** is currently built into the **DN**: each DN output predicts the value of one action (Q-learning). The **Signal Emitters (SE)** are not yet separate: the HC's energy deficit and stomach fill are sensed directly.

### The Simulation Loop
Each tick follows the same sequence:

1. The **SI** converts the current view of the HB into input signals, and the **HC**'s energy deficit and stomach fill are added to them.
2. The **DN** predicts the value of each action. The best action is chosen, except for a small random fraction of steps.
3. The **HB** applies the action. The **HC** updates: food is digested, energy drains, and movement costs extra energy.
4. Reward is computed as the reduction in drive (the squared energy deficit), scaled, plus a novelty bonus from the **NS**.
5. The **DS** compares the reward with the DN's prediction to produce the RPE.
6. The **DN** updates its weights for the action taken, using the RPE.
7. The step is stored in the **MS** with a priority equal to the size of its RPE, and SYNO re-learns from a few past steps, recalled in proportion to their priority (memory replay).

### Hard-Coded vs. Emergent
| Hard-Coded (Physiology) | Emergent (Behavior) |
|---|---|
| Internal variables and set-points | Eating when partially full |
| Energy drain and movement cost | Choosing whether to pursue distant food |
| Signal formulas (hunger, satiety, novelty) | Anticipation when food is seen |
| Reward as drive reduction; RPE | Boredom, resting, and renewed curiosity |
| A small random exploration rate (temporary) | |

### Development Phases
- **Phase 1:** Pure Python, standard library only.
- **Phase 2:** Measured performance bottlenecks are rewritten in Rust.
