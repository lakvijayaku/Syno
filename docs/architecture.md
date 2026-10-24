## Architecture

This document describes how SYNO is structured and how its components interact. Component names are defined in [`glossary.md`](glossary.md). The biological basis for each component is described in [`biology.md`](biology.md).

### Design Principle
SYNO's behavior is never hard-coded. Only the physiology is defined: internal body variables, how they change over time, and the signals released in response. All behavior, including when to eat, explore, or rest, must emerge from learning. See [`ADR 0001`](adr/0001-no-hard-coded-behavior.md).

### Overview
SYNO consists of the **Habitat (HB)**, which is the external world, a body, the **Homeostatic Core (HC)**, and the **Artificial Mind (AM)**, which perceives, decides, and learns. The code is organized the same way:

| Package | Contains |
|---|---|
| `syno/world/` | The Habitat: the grid, food, water, thorns, movement, eating, drinking, and senses |
| `syno/body/` | The Homeostatic Core (energy, stomach, water, fatigue, injury, drive, and reward) and hormones |
| `syno/brain/` | Neurons, layers, networks, training, the Dopamine System, action selection, the Novelty System, the Memory Store, working memory, and the Growth Engine |
| `syno/tools/` | Recording data from runs, smoothing it, and drawing it as charts |
| `experiments/` | Runnable experiments, each adding one mechanism to the one before |
| `tests/` | Unit tests for every module and experiment |

In the diagram below, SYNO's senses pass through the Sensory Interface (SI) to the Decision Network (DN), which chooses an action. Reward goes to the Dopamine System (DS), which compares it with the DN's prediction to produce the reward prediction error (RPE). The Novelty System (NS) adds to the reward, the Memory Store (MS) replays past steps, and the Endocrine Modulator (EM) sets the learning rate. Working memory is not drawn: in the blink experiment, it adds a fading trace of what SYNO saw to the senses that reach the DN.

```
┌──────────────────────── HB ────────────────────────┐
│  food, water, thorns, and open space, and walls    │
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
3. The **HB** applies the action, which may be a move, eating, drinking, or sleeping. The **HC** updates: food is digested, energy drains, movement costs extra energy, and water drains. If energy or water reaches zero, SYNO dies and the life ends.
4. Reward is computed as the reduction in drive (the squared energy deficit, plus the squared water deficit, fatigue, or injury in a body that has one), scaled, plus a novelty bonus from the **Novelty System (NS)**.
5. The **Dopamine System (DS)** compares the reward with the DN's prediction to produce the reward prediction error (RPE).
6. The **DN** updates its weights for the action taken, using the RPE.
7. The step is stored in the **Memory Store (MS)** with a priority equal to the size of its RPE, and SYNO re-learns from a few past steps, recalled in proportion to their priority (memory replay).

In the habituation experiment, SYNO also senses a window of how familiar each nearby square is (`NoveltySystem.sense`), read from the **Novelty System (NS)** after each visit.

In the pain experiment, the world has thorns, which SYNO senses in a second window (`sense_hazard`) along with its injury, and stepping onto a thorn hurts it.

In the sleep experiment, SYNO senses its fatigue and can sleep: it stays still, its fatigue falls, its eyes are closed for the next view, and it gets no novelty bonus. In the dreaming experiment, the replay in step 7 of the loop happens only after a step asleep.

In the blink experiment, SYNO's eyes are closed on some steps, and a working memory of its earlier views is added to its senses in step 1 and updated after each decision.

Across lives, the **Growth Engine (GE)** adds a hidden neuron to the DN whenever SYNO's average surprise stops falling. In the arousal experiment, the arousal hormone of the **Endocrine Modulator (EM)**, fed by surprise, sets the learning rate at the start of each step.

### Hard-Coded vs. Emergent
| Hard-Coded (Physiology) | Emergent (Behavior) |
|---|---|
| Internal variables and set-points | Eating when partially full |
| Energy and water drain, fatigue and recovery, injury and healing, movement cost, and death | Choosing whether to pursue distant food |
| Signal formulas (drive and novelty) | Anticipation when food is seen |
| Reward as drive reduction; reward prediction error (RPE) | Boredom, resting, and renewed curiosity |
| How quickly a square becomes familiar, and fades | Choosing food or water by the larger need |
| | When and how long to sleep |
| A small random exploration rate | Acting on food it can no longer see |
| When the Growth Engine (GE) adds a neuron | |
| How quickly working memory fades | Avoiding thorns, and stepping onto one for food when hungry |
| How many memories are replayed awake and asleep | Leaving a square it has just explored, and returning after time away |
| How much a thorn hurts | |

### Development Phases
- **Phase 1:** Pure Python, standard library only.
- **Phase 2:** Measured performance bottlenecks are rewritten in Rust.
