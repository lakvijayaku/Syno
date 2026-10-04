## Architecture

This document describes how SYNO is structured and how its components interact. Component names are defined in [`glossary.md`](glossary.md). The biological basis for each component is described in [`biology.md`](biology.md).

### Design Principle
SYNO's behavior is never hard-coded. Only the physiology is defined: internal body variables, how they change over time, and the signals released in response. All behavior, including when to eat, explore, or rest, must emerge from learning. See [`ADR 0001`](adr/0001-no-hard-coded-behavior.md).

### Overview
SYNO consists of two parts: the **Habitat (HB)**, which is the external world, and the **Artificial Mind (AM)**, which perceives, decides, and learns.

```
┌──────────────────────── HB ────────────────────────┐
│  food, obstacles, open space                       │
└──────────┬──────────────────────────────▲──────────┘
     senses│                              │action
┌──────────▼─────────────── AM ───────────┴──────────┐
│          SI ─────────────► DN ──────────┘          │
│                            ▲ ▲                     │
│   HC ──► SE ───────────────┘ │ RPE                 │
│   │                          │                     │
│   └──► reward ──► DS ────────┘                     │
│          ▲        ▲ │                              │
│          NS      VE◄┘ RPE                          │
└────────────────────────────────────────────────────┘
```

### The Simulation Loop
Each tick follows the same sequence:

1. The **SI** converts the current view of the HB into input signals.
2. The **SE** report the current hunger and satiety levels from the **HC**.
3. The **DN** selects an action from the sensory and internal signals.
4. The **HB** applies the action. The **HC** updates: energy drains, movement costs extra energy, and eating fills the stomach.
5. Reward is computed as the reduction in deficit, plus a novelty bonus from the **NS**.
6. The **DS** compares the reward against the **VE** prediction to produce the RPE.
7. The **DN** and **VE** update their weights using the RPE.

### Hard-Coded vs. Emergent
| Hard-Coded (Physiology) | Emergent (Behavior) |
|---|---|
| Internal variables and set-points | Eating when partially full |
| Energy drain and movement cost | Choosing whether to pursue distant food |
| Signal formulas (hunger, satiety, novelty) | Anticipation when food is seen |
| Reward as deficit reduction; RPE | Boredom, resting, and renewed curiosity |

### Development Phases
- **Phase 1:** Pure Python, standard library only.
- **Phase 2:** Measured performance bottlenecks are rewritten in Rust.
