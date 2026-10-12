## Architecture

This document describes how SYNO is structured and how its components interact. Component names are defined in [`glossary.md`](glossary.md). The biological basis for each component is described in [`biology.md`](biology.md).

### Design Principle
SYNO's behavior is never hard-coded. Only the physiology is defined: internal body variables, how they change over time, and the signals released in response. All behavior, including when to eat, explore, or rest, must emerge from learning. See [`ADR 0001`](adr/0001-no-hard-coded-behavior.md).

### Overview
SYNO consists of the **Habitat (HB)**, which is the external world, a body, the **Homeostatic Core (HC)**, and the **Artificial Mind (AM)**, which perceives, decides, and learns. The code is organized the same way:

| Package | Contains |
|---|---|
| `syno/world/` | The Habitat: the grid, movement, eating, drinking, and senses |
| `syno/body/` | The Homeostatic Core (energy, stomach, water, drive, and reward) and hormones |
| `syno/brain/` | Neurons, layers, networks, training, the Dopamine System, action selection, the Novelty System, the Memory Store, working memory, and the Growth Engine |
| `syno/tools/` | Recording data from runs, smoothing it, and drawing it as charts |
| `experiments/` | Runnable experiments, each adding one mechanism to the one before |
| `tests/` | Unit tests for every module and experiment |

In the diagram below, SYNO's senses pass through the Sensory Interface (SI) to the Decision Network (DN), which chooses an action. Reward goes to the Dopamine System (DS), which compares it with the DN's prediction to produce the reward prediction error (RPE). The Novelty System (NS) adds to the reward, the Memory Store (MS) replays past steps, and the Endocrine Modulator (EM) sets the learning rate.

```
┌──────────────────────── HB ────────────────────────┐
│  food, water, and open space, bounded by walls     │
└──────────┬──────────────────────────────▲──────────┘
     senses│                              │action
┌──────────▼─────────────── AM ───────────┴──────────┐
│          SI ─────────────► DN ──────────┘          │
│                            ▲ ▲                     │
│   HC ──────────────────────┘ │ RPE                 │
│   │                          │                     │
│   └──► reward ──► DS ────────┘                     │
│          ▲        ▲   │                            │
│          NS       │   ├──► MS ──► replay ──► DN    │
│                   │   └──► EM ──► learning rate    │
│                   └── DN's own value prediction    │
└────────────────────────────────────────────────────┘
```

The **Value Estimator (VE)** is currently built into the **Decision Network (DN)**: each DN output predicts the value of one action (Q-learning). The **Signal Emitters (SE)** are not separate: the HC's energy deficit, stomach fill, and, in a body with water, water deficit are sensed directly.

### The Simulation Loop
Each tick follows the same sequence:

1. The **Sensory Interface (SI)** converts the current view of the Habitat (HB) into input signals: a window of food and walls (`sense`) and, in a world with water, a window of water (`sense_water`). The **Homeostatic Core (HC)**'s energy deficit and stomach fill, and its water deficit if it has water, are added to them.
2. The **Decision Network (DN)** predicts the value of each action. The best action is chosen, except for a small random fraction of steps.
3. The **HB** applies the action, which may be a move, eating, or drinking. The **HC** updates: food is digested, energy drains, movement costs extra energy, and water drains. If energy or water reaches zero, SYNO dies and the life ends.
4. Reward is computed as the reduction in drive (the squared energy deficit, plus the squared water deficit in a body with water), scaled, plus a novelty bonus from the **Novelty System (NS)**.
5. The **Dopamine System (DS)** compares the reward with the DN's prediction to produce the reward prediction error (RPE).
6. The **DN** updates its weights for the action taken, using the RPE.
7. The step is stored in the **Memory Store (MS)** with a priority equal to the size of its RPE, and SYNO re-learns from a few past steps, recalled in proportion to their priority (memory replay).

Across lives, the **Growth Engine (GE)** adds a hidden neuron to the DN whenever SYNO's average surprise stops falling. In the arousal experiment, the arousal hormone of the **Endocrine Modulator (EM)**, fed by surprise, sets the learning rate at the start of each step.

### Hard-Coded vs. Emergent
| Hard-Coded (Physiology) | Emergent (Behavior) |
|---|---|
| Internal variables and set-points | Eating when partially full |
| Energy and water drain, movement cost, and death | Choosing whether to pursue distant food |
| Signal formulas (hunger, satiety, novelty) | Anticipation when food is seen |
| Reward as drive reduction; reward prediction error (RPE) | Boredom, resting, and renewed curiosity |
| | Choosing food or water by the larger need |
| A small random exploration rate | |
| When the Growth Engine (GE) adds a neuron | |

### Development Phases
- **Phase 1:** Pure Python, standard library only.
- **Phase 2:** Measured performance bottlenecks are rewritten in Rust.
