## Biological Basis

This document maps each SYNO component to the biological mechanism it is modeled on. SYNO is inspired by biology, not a replica of it; every simplification is listed explicitly.

### Homeostatic Regulation
The hypothalamus regulates food intake by integrating signals about the body's energy state [1]. Ghrelin, released by the stomach, rises when it is empty and increases hunger. Satiety hormones such as CCK and GLP-1 rise during a meal and reduce hunger gradually rather than instantly. Leptin, released by fat tissue, signals long-term energy stores.

**In SYNO:** The **Homeostatic Core (HC)** holds energy and stomach fill, and digests food into energy gradually. SYNO senses its energy deficit, stomach fill, and, with water, its water deficit directly; separate hunger and satiety signals, the **Signal Emitters (SE)**, are not modeled. Even so, in 7 of 10 seeds a trained SYNO's preference to eat tapers as its stomach fills, so a weak satiety effect emerges from learning alone (signature E1 in [`evaluation.md`](evaluation.md)). `HydratedCore` adds water as a second need: it drains every step, drinking refills it, and the drive is the sum of both squared deficits, so whichever need is larger dominates [6]. SYNO dies when either runs out. In a world with one pond and regrowing food, SYNO learns to keep both needs met: its average lifespan rose from 46 to 88 of 100 steps. With food one step one way and water one step the other, a thirsty SYNO goes for water and a hungry one for food, in 9 of 10 seeds (signature E7).

### Reward Prediction Error
Dopamine neurons encode the difference between received and expected reward [2]. An unexpected reward causes a burst of activity. Once a cue reliably predicts the reward, the burst shifts to the cue, and an omitted reward causes activity to fall below baseline. Dopamine also ramps upward as an animal approaches a distant reward [3].

**In SYNO:** The outputs of the **Decision Network (DN)** predict the value of each action, acting as the **Value Estimator (VE)**. The **Dopamine System (DS)** computes the reward prediction error (RPE) and uses it as the learning signal. At the formula level, the RPE reproduces the burst for unexpected reward, no response for predicted reward, a dip for omitted reward, and a burst at a predictive cue. In behavior, in a world where food can be out of sight and takes time to regrow, SYNO's response moves from eating to the sight of food as it learns: the RPE when eating falls from 2.34 to almost zero, while the RPE when food comes into view rises from −0.28 to +0.18, in every one of 10 seeds (signature E4). A trained SYNO also shows the dip: when food it expects vanishes, its RPE falls to −1.34, against −0.03 when the meal arrives (signature E5). As visible food gets closer, the value SYNO predicts rises at every step, in all 10 seeds, matching the dopamine ramp seen as animals approach a reward, which tracks the value of their situation [15] (signature E9 in [`evaluation.md`](evaluation.md)).

### Wanting vs. Liking
Dopamine drives "wanting," the motivation to pursue a reward, which is separate from "liking," the pleasure of consuming it [4].

**In SYNO:** The Dopamine System (DS) models wanting only. Liking is not modeled.

### State-Dependent Reward
The same stimulus is more rewarding when the body needs it more, a principle called alliesthesia [5]. Homeostatic reinforcement learning formalizes this by defining reward as a reduction in physiological deficit [6].

**In SYNO:** Reward is the reduction in drive. For a body with only energy (`HomeostaticCore`), drive is the squared energy deficit; for a body that also needs water (`HydratedCore`), it is the sum of the squared energy and water deficits; and for a body that gets tired (`FatiguedCore`), it is the sum of the squared energy deficit and the squared fatigue. Squaring makes the same food worth more to a hungrier SYNO: digestion that is worth 0.014 when nearly full is worth 0.062 when hungry. In experiments, a trained SYNO chooses to eat whenever it is hungry, including when partially full. However, it also prefers to eat at almost any energy, so its willingness to eat barely changes with hunger, unlike an animal's (signature E2 in [`evaluation.md`](evaluation.md)), and a nearly full SYNO pursues distant food as readily as nearby food (signature E3).

### Novelty and Habituation
Novel stimuli elicit a response that weakens with repeated exposure [7].

**In SYNO:** The **Novelty System (NS)** provides a novelty bonus that habituates with repetition and recovers over time. SYNO cannot sense how familiar a square is, so habituation shapes its learning but cannot be measured in its behavior (signature E6 in [`evaluation.md`](evaluation.md)).

### Exploration and Exploitation
The locus coeruleus–norepinephrine system adjusts whether an animal focuses on a known reward or disengages to explore [8].

**In SYNO:** The **Arousal System (AS)** is not modeled. Exploration comes from the **Novelty System (NS)** plus a small random exploration rate. The novelty bonus let the random rate fall from 30% to 5%, but without any randomness, SYNO did not learn.

### Memory Replay
During rest and sleep, the hippocampus replays sequences of recent experience, which helps consolidate them into long-term memory [9]. Replay is not uniform: experiences linked to reward are replayed more often [10], and replay patterns match a model that favors the memories most useful to learn from [11].

**In SYNO:** The **Memory Store (MS)** holds SYNO's most recent experiences and forgets the oldest when full. After every step, SYNO re-learns from 2 recalled experiences. Recalling at random, SYNO reached an average energy of about 0.86 roughly three times sooner, and its worst dip after learning was smaller (0.76 rather than 0.64). Recalling in proportion to surprise, measured as the size of each memory's reward prediction error (RPE), SYNO learned faster still, and its worst dip after learning rose to 0.86. In the sleep experiment, replaying 6 memories after each step asleep and none while awake, about 1.8 per step on average, worked as well as replaying 2 after every step: SYNO's average energy rose from 0.32 to 0.69 over 3000 lives, while it still slept 30% of the time.

### Working Memory
Some neurons in the prefrontal cortex keep firing after a stimulus disappears, holding its location for several seconds until the animal acts [14]. In the delayed-response task, an animal sees where food is hidden, waits, and then must find it, which depends on this persistent activity.

**In SYNO:** Working memory is a fading trace of SYNO's view: at every step it keeps half of what it held and takes in the rest from what SYNO saw. In a world where SYNO's eyes are closed on half of all steps, a SYNO that saw food just before its eyes closed moves toward it in 0.75 of cases, against 0.56 when it never saw the food, in 9 of 10 seeds (signature E8 in [`evaluation.md`](evaluation.md)). The memory fades within a few steps, much faster than an animal's.

### Sleep and Fatigue
The need for sleep builds up the longer an animal is awake and is discharged by sleeping; this sleep pressure explains both why animals fall asleep more readily when tired and why they sleep longer after being kept awake (sleep rebound) [16]. In the brain, adenosine accumulates during wakefulness and promotes sleep [17].

**In SYNO:** `FatiguedCore` adds fatigue as a second need: it builds up on every waking step, like adenosine, and falls on every step asleep, and sleep burns half the energy of being awake. The drive adds the squared fatigue to the squared energy deficit. In the scarce world with sleep as a sixth action, SYNO learns to sleep about 30% of the time, close to the 29% that keeps fatigue level, while its average energy rose from 0.32 to 0.70 over 3000 lives. With no food in sight, a tired SYNO chooses sleep more often than a rested one in 8 of 10 seeds (signature E10 in [`evaluation.md`](evaluation.md)), and once asleep, a more tired SYNO sleeps about 15 steps rather than 5, in all 10 seeds (signature E11). No rule says when or how long to sleep.

### Neurogenesis
New neurons continue to be born in parts of the adult brain, including the human hippocampus [12], and must integrate into existing circuits without disrupting them.

**In SYNO:** The **Growth Engine (GE)** adds a neuron to a hidden layer of the **Decision Network (DN)**. The new neuron's outgoing connections start at zero, so the DN's output is unchanged at first, and the connections are then learned. A network too small to learn XOR solved it after growing two neurons. In the habitat, SYNO starts with one hidden neuron and grows one whenever its average surprise stops falling. It took off once it reached 5 to 6 neurons, matching the performance of the 8-neuron network that was sized by hand. Surprise also stops falling once the task is mastered, so SYNO keeps growing after learning is complete, up to a fixed limit.

### Hormones and Arousal
Hormones and neuromodulators act slowly and globally, so their levels reflect the recent past rather than the present moment. In the Pearce-Hall model of learning, attention to a situation, and so the speed of learning about it, rises after surprising outcomes and falls as outcomes become predictable [13].

**In SYNO:** A hormone's level drifts a fixed fraction of the way toward its signal on every step. An arousal hormone fed by surprise (the size of the reward prediction error, or RPE) sets SYNO's learning rate. When SYNO moved from a 3x3 world to a 5x5 world, arousal raised its average energy over the next 250 lives from 0.46 to 0.65 for seed 0, and from 0.46 to 0.55 averaged over 6 seeds, with 5 of 6 seeds improving; over the longer run, SYNO performed about the same with or without it. A stress hormone fed by hunger that raised random exploration was also tested, and made SYNO worse after the move. These hormones are numbers that change how SYNO learns. They do not imply feelings.

### Simplifications
- Hormones are single numbers, not concentrations that diffuse through tissue.
- The hypothalamus, ventral tegmental area (VTA), and nucleus accumbens are not modeled as separate brain regions.
- Leptin and long-term energy storage are not modeled.
- Sleep is a single action, with no sleep stages and no day-night (circadian) cycle.
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
10. Ambrose, R. E., Pfeiffer, B. E., & Foster, D. J. (2016). Reverse replay of hippocampal place cells is uniquely modulated by changing reward. *Neuron*, 91(5), 1124–1136.
11. Mattar, M. G., & Daw, N. D. (2018). Prioritized memory access explains planning and hippocampal replay. *Nature Neuroscience*, 21(11), 1609–1617.
12. Eriksson, P. S., Perfilieva, E., Björk-Eriksson, T., Alborn, A.-M., Nordborg, C., Peterson, D. A., & Gage, F. H. (1998). Neurogenesis in the adult human hippocampus. *Nature Medicine*, 4(11), 1313–1317.
13. Pearce, J. M., & Hall, G. (1980). A model for Pavlovian learning: Variations in the effectiveness of conditioned but not of unconditioned stimuli. *Psychological Review*, 87(6), 532–552.
14. Funahashi, S., Bruce, C. J., & Goldman-Rakic, P. S. (1989). Mnemonic coding of visual space in the monkey's dorsolateral prefrontal cortex. *Journal of Neurophysiology*, 61(2), 331–349.
15. Hamid, A. A., Pettibone, J. R., Mabrouk, O. S., Hetrick, V. L., Schmidt, R., Vander Weele, C. M., Kennedy, R. T., Aragona, B. J., & Berke, J. D. (2016). Mesolimbic dopamine signals the value of work. *Nature Neuroscience*, 19(1), 117–126.
16. Borbély, A. A. (1982). A two process model of sleep regulation. *Human Neurobiology*, 1(3), 195–204.
17. Porkka-Heiskanen, T., Strecker, R. E., Thakkar, M., Bjørkum, A. A., Greene, R. W., & McCarley, R. W. (1997). Adenosine: A mediator of the sleep-inducing effects of prolonged wakefulness. *Science*, 276(5316), 1265–1268.
