## Glossary

This glossary defines every named component and key term used in SYNO. Component names are deliberately short; the expanded name describes exactly what each component does.

### Components
| Name | Full Name | Role | Status |
|---|---|---|---|
| **AM** | Artificial Mind | The complete cognitive system. Owns and orchestrates every component below. | In progress |
| **HB** | Habitat | The grid world SYNO lives in, containing food, water, and open space, bounded by walls. External to the AM. | Built: `syno/world/habitat.py` |
| **SI** | Sensory Interface | Converts what SYNO can perceive in the HB into numeric input signals. | Built: `Habitat.sense` |
| **DN** | Decision Network | The neural network that maps sensory and internal signals to an action. | Built: `syno/brain/` (one value per action) |
| **HC** | Homeostatic Core | Holds SYNO's internal body variables (energy, stomach fill) and updates them every tick. | Built: `syno/body/homeostasis.py`. `HydratedCore` adds water, a second need, and death when energy or water runs out |
| **SE** | Signal Emitters | Hormone-like signals computed from the HC: the hunger signal (ghrelin analog) and the satiety signal (CCK/GLP-1 analog). | Partial: deficit and stomach fill are sensed directly |
| **VE** | Value Estimator | Predicts how much reward SYNO expects from its current situation. | Built into the DN: each output predicts the value of one action |
| **DS** | Dopamine System | Computes the reward prediction error (actual reward minus expected reward) and broadcasts it as the learning signal. | Built: `syno/brain/dopamine.py` |
| **NS** | Novelty System | Produces a novelty signal for unfamiliar states, which fades with repeated exposure. | Built: `syno/brain/novelty.py` |
| **AS** | Arousal System | Norepinephrine analog that tunes the balance between exploring and exploiting. | Not modeled. The arousal hormone in `experiments/arousal.py` sets the learning rate, not exploration |
| **MS** | Memory Store | Records experiences for replay, forgetting, and consolidation. | Built: `syno/brain/memory.py`, with uniform replay (`experiments/replay.py`) and surprise-weighted replay (`experiments/consolidation.py`) |
| **GE** | Growth Engine | Adds neurons to the DN over time, without changing what the DN outputs. | Built: `syno/brain/growth.py`. In `experiments/neurogenesis.py`, SYNO grows when its surprise stops falling |
| **EM** | Endocrine Modulator | Slow, global hormone and emotion analogs that modulate all other components. | Built: `syno/body/hormones.py`. In `experiments/arousal.py`, an arousal hormone fed by surprise sets the learning rate |

### Terms
- **Tick:** One step of the simulation loop. SYNO senses, acts, and learns once per tick.
- **Homeostasis:** The tendency of a body to keep internal variables near a set-point.
- **Deficit:** The distance between an internal variable and its set-point. Larger deficit means stronger need.
- **Reward:** In SYNO, a reduction in drive, plus a novelty bonus. Reward is never assigned directly to a behavior.
- **Drive:** The squared energy deficit. Reward is the reduction in drive over a step.
- **Q-Learning:** The learning method SYNO uses: each output of the DN predicts the value of one action, and only the action taken is updated by its RPE.
- **Epsilon-Greedy:** Choosing the best-valued action, except for a random action a small fraction (epsilon) of the time.
- **Life:** One run of an experiment, from SYNO's first step to its last.
- **Reward Prediction Error (RPE):** Actual reward minus expected reward. Positive when better than expected, negative when worse.
- **Alliesthesia:** The principle that the same stimulus is more rewarding when the body needs it more.
- **Habituation:** The gradual weakening of a response to a repeated stimulus.
- **Exploration / Exploitation:** Trying unfamiliar actions versus repeating actions known to be rewarding.
- **Learning Rate:** How large a change each RPE makes to the DN's weights.
- **Experience Replay:** Re-learning from past steps recalled from the MS, in addition to the current step.
- **Prioritized Replay:** Experience replay that recalls each memory in proportion to its surprise, measured as the size of its RPE.
- **Neurogenesis:** Adding new neurons to a network that is already learning. In SYNO, this is done by the GE.
- **Hormone:** A slow signal whose level moves a fixed fraction of the way toward its input on every tick, so it reflects the recent past.
- **Arousal:** In SYNO, a hormone fed by surprise that raises the learning rate after surprising outcomes (the Pearce-Hall model).
- **Seed:** The starting value for random numbers. The same seed reproduces the same run exactly.
- **Behavioral Signature:** A behavior documented in biological research that SYNO should reproduce without it being hard-coded. Listed in [`evaluation.md`](evaluation.md).
- **Emergent Behavior:** Behavior that arises from the interaction of components rather than being explicitly programmed.
