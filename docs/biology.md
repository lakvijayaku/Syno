## Biological Basis

This document maps each SYNO component to the biological mechanism it is modeled on. SYNO is inspired by biology, not a replica of it; every simplification is listed explicitly.

### Homeostatic Regulation
The hypothalamus regulates food intake by integrating signals about the body's energy state [1]. Ghrelin, released by the stomach, rises when it is empty and increases hunger. Satiety hormones such as CCK and GLP-1 rise during a meal and reduce hunger gradually rather than instantly. Leptin, released by fat tissue, signals long-term energy stores.

**In SYNO:** The **HC** holds energy and stomach fill, and digests food into energy gradually. SYNO senses its energy deficit and stomach fill directly; separate hunger and satiety signals (**SE**) are not yet modeled.

### Reward Prediction Error
Dopamine neurons encode the difference between received and expected reward [2]. An unexpected reward causes a burst of activity. Once a cue reliably predicts the reward, the burst shifts to the cue, and an omitted reward causes activity to fall below baseline. Dopamine also ramps upward as an animal approaches a distant reward [3].

**In SYNO:** The **DN**'s outputs predict the value of each action, acting as the **VE**. The **DS** computes the RPE and uses it as the learning signal. At the formula level, the RPE reproduces the burst for unexpected reward, no response for predicted reward, a dip for omitted reward, and a burst at a predictive cue.

### Wanting vs. Liking
Dopamine drives "wanting," the motivation to pursue a reward, which is separate from "liking," the pleasure of consuming it [4].

**In SYNO:** The DS models wanting only. Liking is not modeled.

### State-Dependent Reward
The same stimulus is more rewarding when the body needs it more, a principle called alliesthesia [5]. Homeostatic reinforcement learning formalizes this by defining reward as a reduction in physiological deficit [6].

**In SYNO:** Reward is the reduction in drive, where drive is the squared energy deficit. Squaring makes the same food worth more to a hungrier SYNO: digestion that is worth 0.014 when nearly full is worth 0.062 when hungry. In experiments, a trained SYNO chooses to eat whenever it is hungry, including when partially full.

### Novelty and Habituation
Novel stimuli elicit a response that weakens with repeated exposure [7].

**In SYNO:** The **NS** provides a novelty bonus that habituates with repetition and recovers over time.

### Exploration and Exploitation
The locus coeruleus–norepinephrine system adjusts whether an animal focuses on a known reward or disengages to explore [8].

**In SYNO:** The **AS** is not modeled. Exploration comes from the **NS** plus a small random exploration rate. Without any randomness, SYNO did not learn.

### Memory Replay
During rest and sleep, the hippocampus replays sequences of recent experience, which helps consolidate them into long-term memory [9].

**In SYNO:** The **MS** holds SYNO's most recent experiences and forgets the oldest when full. Experiences are recalled at random rather than in order.

### Simplifications
- Hormones are single numbers, not concentrations that diffuse through tissue.
- The hypothalamus, VTA, and nucleus accumbens are not modeled as separate brain regions.
- Leptin and long-term energy storage are not modeled in the first version.
- "Liking" and conscious experience are not modeled. SYNO's signals are numbers; they do not imply feelings.

### References
1. Morton, G. J., Cummings, D. E., Baskin, D. G., Barsh, G. S., & Schwartz, M. W. (2006). Central nervous system control of food intake and body weight. *Nature*, 443(7109), 289–295.
2. Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593–1599.
3. Howe, M. W., Tierney, P. L., Sandberg, S. G., Phillips, P. E. M., & Graybiel, A. M. (2013). Prolonged dopamine signalling in striatum signals proximity and value of distant rewards. *Nature*, 500(7464), 575–579.
4. Berridge, K. C., & Robinson, T. E. (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *Brain Research Reviews*, 28(3), 309–369.
5. Cabanac, M. (1971). Physiological role of pleasure. *Science*, 173(4002), 1103–1107.
6. Keramati, M., & Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. *eLife*, 3, e04811.
7. Thompson, R. F., & Spencer, W. A. (1966). Habituation: A model phenomenon for the study of neuronal substrates of behavior. *Psychological Review*, 73(1), 16–43.
8. Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus coeruleus–norepinephrine function: Adaptive gain and optimal performance. *Annual Review of Neuroscience*, 28, 403–450.
9. Wilson, M. A., & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*, 265(5172), 676–679.
