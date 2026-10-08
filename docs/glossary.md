## Glossary

This glossary defines every named component and key term used in SYNO. Component names are deliberately short; the expanded name describes exactly what each component does.

### Components
| Name | Full Name | Role | Status |
|---|---|---|---|
| **AM** | Artificial Mind | The complete cognitive system. Owns and orchestrates every component below. | In progress |
| **HB** | Habitat | The grid world SYNO lives in, containing food, obstacles, and open space. External to the AM. | Built: `syno/world/habitat.py` |
| **SI** | Sensory Interface | Converts what SYNO can perceive in the HB into numeric input signals. | Built: `Habitat.sense` |
| **DN** | Decision Network | The neural network that maps sensory and internal signals to an action. | Built: `syno/brain/` (one value per action) |
| **HC** | Homeostatic Core | Holds SYNO's internal body variables (energy, stomach fill) and updates them every tick. | Built: `syno/body/homeostasis.py` |
| **SE** | Signal Emitters | Hormone-like signals computed from the HC: the hunger signal (ghrelin analog) and the satiety signal (CCK/GLP-1 analog). | Partial: deficit and stomach fill are sensed directly |
| **VE** | Value Estimator | Predicts how much reward SYNO expects from its current situation. | Built into the DN: each output predicts the value of one action |
| **DS** | Dopamine System | Computes the reward prediction error (actual reward minus expected reward) and broadcasts it as the learning signal. | Built: `syno/brain/dopamine.py` |
| **NS** | Novelty System | Produces a novelty signal for unfamiliar states, which fades with repeated exposure. | Built: `syno/brain/novelty.py` |
| **AS** | Arousal System | Norepinephrine analog that tunes the balance between exploring and exploiting. | Planned |
| **MS** | Memory Store | Records experiences for replay, forgetting, and consolidation. | Planned |
| **GE** | Growth Engine | Adds neurons and connections to the DN over time (NEAT). | Planned |
| **EM** | Endocrine Modulator | Slow, global hormone and emotion analogs that modulate all other components. | Planned |

### Terms
- **Tick:** One step of the simulation loop. SYNO senses, acts, and learns once per tick.
- **Homeostasis:** The tendency of a body to keep internal variables near a set-point.
- **Deficit:** The distance between an internal variable and its set-point. Larger deficit means stronger need.
- **Reward:** In SYNO, a reduction in deficit. Reward is never assigned directly to a behavior.
- **Reward Prediction Error (RPE):** Actual reward minus expected reward. Positive when better than expected, negative when worse.
- **Alliesthesia:** The principle that the same stimulus is more rewarding when the body needs it more.
- **Habituation:** The gradual weakening of a response to a repeated stimulus.
- **Exploration / Exploitation:** Trying unfamiliar actions versus repeating actions known to be rewarding.
- **Emergent Behavior:** Behavior that arises from the interaction of components rather than being explicitly programmed.
