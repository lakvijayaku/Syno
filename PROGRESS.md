## Progress

### Current Status
Phase 1 is complete: all 20 roadmap steps are done. Measured over 10 seeds, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, and E14 are reproduced, E1 weakly, and E2 and E3 are not, and SYNO has 30% of the human capabilities listed in `docs/humanness.md`. Phase 2 rewrites SYNO in Rust on the `rust_master` branch (`docs/adr/0002-rewrite-in-rust.md`).

### Roadmap
| Step | Milestone | Status |
|---|---|---|
| **1** | Single neuron | ✅ Done |
| **2a** | Layer | ✅ Done |
| **2b** | Network | ✅ Done |
| **2c** | Loss | ✅ Done |
| **2d** | Gradient descent on a single neuron | ✅ Done |
| **2e** | Backpropagation | ✅ Done |
| **2f** | Train on XOR | ✅ Done |
| **3** | Habitat (grid world) | ✅ Done |
| **4** | Dopamine as reward prediction error | ✅ Done |
| **5** | Drives | ✅ Done |
| **6** | Logging and plots | ✅ Done |
| **7** | Memory | ✅ Done |
| **8** | Neuron growth | ✅ Done |
| **9** | Hormones and emotions | ✅ Done |
| **10** | Evaluation signatures | ✅ Done |
| **11** | A harder world | ✅ Done |
| **12** | Trustworthy measurement | ✅ Done |
| **13** | Thirst | ✅ Done |
| **14** | Working memory | ✅ Done |
| **15** | Dopamine ramping | ✅ Done |
| **16** | Fatigue and sleep | ✅ Done |
| **17** | Consolidation during sleep | ✅ Done |
| **18** | Pain and fear | ✅ Done |
| **19** | Habituation in behavior | ✅ Done |
| **20** | Phase 1 wrap-up | ✅ Done |

### Log
#### 2026-10-23
- **Step 20 complete:** every slow test passed, so every recorded result reproduces from the code. The Signal Emitters (7 designs) and an Arousal System that sets exploration (6 designs, over 4 seeds each) were tested and ruled out, and `docs/biology.md` records the evidence. ADR 0002 records the decision to rewrite SYNO in Rust on the `rust_master` branch. Injury and healing becomes a partial SYNO capability, which step 18 had missed, raising the humanness score to 30%. Every test helper now has a docstring. Phase 1 is complete.
- **Step 19 complete:** SYNO senses how familiar each nearby square is with `NoveltySystem.sense`, in `experiments/habituation.py`, and `experiments/signatures.py` measures E6, habituation, on a SYNO from each seed trained there, covered by 30 new tests. Over 3000 lives, SYNO's average energy rose from 0.506 to 0.685. With no food in sight, it steps onto a neighboring square in 0.31 of cases when it is new, 0.01 after visiting it 4 times, and 0.30 after 80 steps away, in all 10 seeds, so E6, recorded as not measurable since step 10, is reproduced. Habituation becomes a SYNO capability, raising the humanness score to 29.5%. Step 19 (Habituation in behavior) is complete.

#### 2026-10-18
- **Readability:** `Neuron.forward` computes its weighted sum with a plain loop instead of a one-line generator, checks the input count itself, and adds the bias last, so every result is unchanged. The loop also runs about a third faster.

#### 2026-10-15
- **Step 18d complete:** `experiments/signatures.py` measures E13, pain avoidance, and E14, hunger overrides pain, on a SYNO from each seed trained in the pain experiment, covered by 13 new tests. With no food in sight, SYNO steps onto a neighboring square in 0.29 of cases when it is empty and 0.00 when it has a thorn, and with food growing on the thorn, it steps onto it in 0.72 of cases at energy 0.2 against 0.29 at 0.8, both in all 10 seeds, so both are reproduced. Fear and threat avoidance becomes a partial SYNO capability, raising the humanness score to 29%. Step 18 (Pain and fear) is complete.
- **Step 18c complete:** `experiments/pain.py`, where the scarce world has 2 thorns and stepping onto one hurts SYNO, covered by 16 tests. Food can grow on a thorn. Over 3000 lives, SYNO's average energy rose from 0.25 to 0.57, and it stepped onto a thorn about 0.6 times per life. Pain signals become a SYNO capability, raising the humanness score to 28.5%.
- **Step 18b complete:** `InjuredCore`, a body with injury as a second need, covered by 24 unit tests. Injury rises when SYNO is hurt, heals by 0.02 every step, and adds its square to the drive, so being hurt is punished and healing rewarded. `HomeostaticCore` is unchanged.
- **Step 18a complete:** the Habitat can hold thorns, which SYNO can step onto and see with `sense_hazard`, covered by 20 new unit tests. Worlds without thorns are unchanged.
- **Documentation:** `CLAUDE.md` says the commit command comes only after every fix is in and the tests pass.
- **Step 17c complete:** `experiments/signatures.py` measures E12, sleep consolidation, by training two dreaming SYNOs from each seed, one with its sleep replay blocked, covered by 8 new tests. Over its last 500 lives, the SYNO that replays during sleep reaches an average energy of 0.696, against 0.318 with sleep replay blocked, in all 10 seeds, so E12 is reproduced. Consolidation during sleep becomes a SYNO capability, raising the humanness score to 27.5%. Step 17 (Consolidation during sleep) is complete.
- **Step 17b complete:** `experiments/dreaming.py`, where SYNO replays 6 memories after each step asleep and none while awake, covered by 8 tests. Its average energy rose from 0.32 to 0.69 over 3000 lives, as high as when it replays after every step, and it still slept 30% of the time.
- **Step 17a complete:** `sleep.live` takes how many memories SYNO replays after a step awake and after a step asleep, covered by 6 new tests. Both default to 2, so the sleep experiment and its results are unchanged.

#### 2026-10-14
- **Step 16c complete:** `experiments/signatures.py` measures E10, sleep pressure, and E11, sleep rebound, on a SYNO from each seed trained in the sleep experiment, covered by 17 new tests. With no food in sight, SYNO chooses to sleep on 0.68 of squares at fatigue 0.2 and on all of them at 0.8, in 8 of 10 seeds, and once asleep it sleeps 4.98 steps at fatigue 0.4 against 15.03 at 0.9, in all 10 seeds, so both are reproduced. The humanness score stays at 26.5%. Step 16 (Fatigue and sleep) is complete.
- **Step 16b complete:** `experiments/sleep.py`, where SYNO forages in the scarce world with sleep as a sixth action, covered by 21 tests. While asleep it stays still, its eyes are closed, and it gets no novelty bonus. Over 3000 lives its average energy rose from 0.32 to 0.70, and its time asleep fell from 0.44 to 0.30, close to the 0.29 that keeps fatigue level. Fatigue and rest become a SYNO capability, and sleep a partial one, raising the humanness score to 26.5%.
- **Step 16a complete:** `FatiguedCore`, a body with fatigue as a second need, covered by 28 unit tests. Fatigue builds up on every waking step and falls on every sleeping step, sleep burns half the base energy, and the drive adds the squared fatigue to the squared energy deficit. `HomeostaticCore` is unchanged, so earlier experiments are unaffected.
- **Documentation:** `CLAUDE.md` records how Claude Code tutors on SYNO, including catching up from the repo at the start of each session, the commit message style, and the checks before every commit. The README links to it, and `Neuron`, `Layer`, and `Network` gained class docstrings.

#### 2026-10-12
- **Step 15 complete:** `experiments/signatures.py` measures E9, dopamine ramping, on the scarce-world SYNO, covered by 6 new tests. As visible food gets closer, the value SYNO predicts rises at every step, from 1.197 at 4 steps away to 2.288 on the food, in all 10 seeds, so E9 is reproduced. The humanness score stays at 25%, since dopamine as prediction error was already a SYNO capability.

#### 2026-10-11
- **Documentation:** every document checked against the code after steps 13 and 14. The README, glossary, architecture, biology, and humanness now cover water, the water senses, lifespans, and working memory throughout; the glossary's Life term and Signal Emitters status were corrected.
- **Step 14c complete:** `experiments/signatures.py` measures E8, delayed response, on a SYNO from each seed trained in the blink experiment, covered by 7 new tests. A SYNO that saw food just before its eyes closed moves toward it in 0.75 of cases, against 0.56 when it never saw it, in 9 of 10 seeds, so E8 is reproduced. Working memory becomes a partial SYNO capability, raising the humanness score to 25%. Step 14 (Working memory) is complete.
- **Step 14b complete:** `experiments/blink.py`, where SYNO's eyes are closed on half of all steps and it senses a working memory of what it saw before, covered by 15 tests. Its average energy rose from 0.27 to 0.49 over 3000 lives.
- **Step 14a complete:** `WorkingMemory`, a fading trace of recent inputs that keeps a fraction of its old contents at every update, covered by 18 unit tests.

#### 2026-10-10
- **Tooling:** a GitHub Actions workflow runs the fast tests on every push and pull request, on Python 3.10 and 3.13, and the README shows its result as a badge.
- **Documentation:** biology and architecture describe the two-need drive, the water senses, and death; abbreviations are written in full on first use in each section; and CONTRIBUTING requires any commit that makes a document wrong to fix it in the same commit.
- **Step 13d complete:** `experiments/signatures.py` measures E7, need switching, on a SYNO from each seed trained in the thirst experiment, covered by 8 new tests. With food and water on opposite sides, 9 of 10 seeds go for whichever they need more, so E7 is reproduced and need-dependent choice becomes a SYNO capability, raising the humanness score to 24.5%. Step 13 (Thirst) is complete.

#### 2026-10-09
- **Step 13c complete:** `experiments/thirst.py`, where SYNO must keep itself fed and watered on the 5x5 grid with one pond, and dies if either runs out, covered by 17 tests. With 16 hidden neurons, its average lifespan rose from 45.7 to 87.5 of 100 steps over 3000 lives. Thirst and death from neglected needs are now SYNO capabilities, raising the humanness score from 22% to 24%.
- **Step 13b complete:** the Habitat can hold water, which SYNO can drink without using it up and see with `sense_water`, covered by 17 new unit tests. Worlds without water are unchanged.
- **Step 13a complete:** `HydratedCore`, a body with water as a second need, covered by 30 unit tests. Water drains every step and is refilled by drinking, the drive adds both squared deficits, and SYNO dies when energy or water runs out. `HomeostaticCore` is unchanged, so earlier experiments are unaffected.
- **Step 12 complete:** `experiments/signatures.py` measures 10 seeds in parallel (about 90 seconds), reports each result's range, and counts how many seeds show each signature, covered by 55 tests. A signature is reproduced when 8 or more seeds show it: E4 and E5 (10 of 10), E1 weakly (7 of 10), E2 (2 of 10) and E3 (1 of 10) not. Every status, and the 22% score, is unchanged.
- **Documentation:** `docs/humanness.md` now scores SYNO against 104 human capabilities rather than the 6 measured signatures. SYNO has 17, has 10 in part, and lacks 77; the 4 that cannot be tested even in animals are left out, giving a score of 22%.

#### 2026-10-08
- **Step 11b complete:** `experiments/signatures.py` measures SYNO trained in the scarce world, and every probe now tests every position, so measuring uses no random numbers, covered by 46 tests. E4 is now fully reproduced: the RPE when food comes into view rises from −0.173 to +0.174. E5's dip deepened to −1.305, and E1 tapers more gradually. The humanness score rose from 33% to 42%. Step 11 (A harder world) is complete.
- **Step 11a complete:** `experiments/scarcity.py`, where SYNO forages on a 5x5 grid, larger than its view, and eaten food takes 10 steps to regrow, covered by 13 tests. Average energy rose from 0.30 to 0.66 over 3000 lives. Every world tried took about the same time per life, so the larger world needs no Rust.
- **Documentation:** `docs/humanness.md` scores SYNO at 33% (2 of 6 points) from the measured signatures, and lists what earns the score and what holds it back.
- **Step 10f complete:** `approach_rate` measures E3, partial-fullness eating, covered by 9 new tests. A nearly full SYNO moves toward distant food as often as toward adjacent food (0.92 for both), so E3 is recorded as not reproduced. Step 10 (Evaluation signatures) is complete, and with it the roadmap.
- **Step 10e complete:** `eating_rate` takes a stomach fill and measures E1, satiation, covered by 5 new tests. A half-hungry SYNO prefers to eat at every stomach fill up to 0.4, then less often (0.96, 0.91, and 0.71 at full). E1 is recorded as weakly reproduced.
- **Step 10d complete:** E6, habituation, is recorded as not measurable with the current design, because SYNO cannot sense how familiar a square is and the 3x3 grid has no unfamiliar area to return to.
- **Step 10c complete:** `cue_rpe` measures E4, RPE transfer, at 4 points in training, using `new_brain` and a `train` that can be called in parts, covered by 11 new or rewritten tests. The RPE when eating falls from 2.443 to −0.083, but the RPE when food appears never becomes positive, because food on the 3x3 grid is always in view. E4 is recorded as partly reproduced.
- **Step 10b complete:** `eating_rpe` measures E5, reward omission, covered by 7 new tests. When food SYNO expects vanishes just before it eats, its RPE averages −0.874 across 5 seeds, against −0.083 when the meal arrives. E5 is recorded as reproduced.
- **Step 10a complete:** `experiments/signatures.py` trains SYNO with 5 seeds and measures E2, state-dependent eating, covered by 15 tests. SYNO prefers to eat at every energy from 0.0 to 0.9 and in 73% of cases at full energy, so its eating does not rise gradually with deficit. E2 is recorded as not reproduced.
- Documentation brought up to the current state: README (status, running SYNO), architecture (packages, diagram), glossary (AS status, new terms), biology, and PROGRESS (step 10).
- **Step 9b complete:** `experiments/arousal.py`, where an arousal hormone fed by surprise sets the learning rate, covered by 10 tests. After moving to a larger world, SYNO recovered faster (average energy 0.65 rather than 0.46 over the next 250 lives) but performed about the same in the long run. A stress hormone that raised exploration was tested and rejected. `spawn_food` now works on any grid size. Step 9 (Hormones and emotions) is complete, and with it the roadmap.

#### 2026-10-07
- **Step 9a complete:** `Hormone`, a slow signal that builds up and fades gradually, covered by 11 unit tests.
- **Step 8b complete:** `experiments/neurogenesis.py`, where SYNO starts with one hidden neuron and grows whenever its surprise stops falling, covered by 8 tests. It reached an average energy of 0.93, matching the hand-sized network. Step 8 (Neuron growth) is complete.
- **Step 8a complete:** `grow_neuron` adds a neuron to a hidden layer without changing the network's output, covered by 14 unit tests.
- **Step 7c complete:** priority-weighted recall in `MemoryStore`, and `experiments/consolidation.py`, where surprising memories are replayed most often, covered by 18 tests. Average energy reached 0.72 in the first 500 lives, and the worst dip after learning rose to 0.86. Step 7 (Memory) is complete.
- **Step 7b complete:** `experiments/replay.py`, where SYNO replays 2 random memories after every step, covered by 7 tests. Average energy reached about 0.86 roughly three times sooner than without replay, and stayed between 0.90 and 0.93.
- **Step 7a complete:** `MemoryStore` holds recent experiences and forgets the oldest, covered by 11 unit tests.
- **Step 6c complete:** `moving_average`, and the curiosity experiment now saves every life to `runs/curiosity.csv` and a chart to `runs/curiosity.svg`, covered by 9 unit tests. Step 6 (Logging and plots) is complete.
- **Step 6b complete:** `line_chart` draws data as an SVG chart, covered by 12 unit tests.
- **Step 6a complete:** `Recorder` saves run data as CSV, covered by 18 unit tests.
- Documentation brought up to date: README status and AI policy, architecture overview and loop, biology, evaluation statuses, and glossary terms.
- **Step 5e complete:** `NoveltySystem` with habituation and recovery, and `experiments/curiosity.py`, covered by 21 tests. Curiosity let SYNO learn with 5% random exploration instead of 30%, reaching an average energy of 0.86 to 0.91. Step 5 (Drives) is complete.
- Glossary statuses updated for the components built so far.
- **Step 5d complete:** `experiments/hunger.py`, where SYNO keeps itself fed with reward coming only from its body, covered by 7 tests. Average energy over a life rose from 0.23 to 0.82, and SYNO eats whenever hungry, including when half-full.
- **Step 5c complete:** linear activation for neurons, with training using each neuron's own slope, covered by 15 new unit tests.
- **Step 5b complete:** drive and `homeostatic_reward`, so reward comes from reducing hunger, covered by 9 unit tests.
- **Step 5a complete:** `HomeostaticCore` with energy, digestion, and stomach capacity, covered by 20 unit tests.
- **Step 4d complete:** `experiments/forage.py`, where SYNO learns to find and eat food from its own RPE, covered by 9 tests. On a 3x3 grid, the average steps to eat fell from 38.7 to 6.1. Step 4 (Dopamine) is complete.
- **Step 4c complete:** `learn` Q-learning update driven by the RPE, covered by 15 unit tests.
- **Step 4b complete:** `choose_action` epsilon-greedy policy, covered by 11 unit tests.
- **Step 4a complete:** `reward_prediction_error`, the Dopamine System, covered by 14 unit tests.
- **Step 3d complete:** senses report a window around SYNO as numbers, covered by 13 unit tests. Step 3 (Habitat) is complete.
- **Step 3c complete:** eating removes food from SYNO's square, covered by 7 unit tests.
- **Step 3b complete:** movement actions with grid edges acting as walls, covered by 12 unit tests.
- **Step 3a complete:** `Habitat` grid with bounds checking and text rendering, covered by 16 unit tests.
- **Step 2f complete:** `experiments/xor.py` trains a 2-3-1 network to solve XOR, covered by 7 tests.
- **Step 2e complete:** `train_network_step` backpropagation for multi-layer networks, covered by 14 unit tests.
- README project status updated to reflect current progress.

#### 2026-10-06
- **Step 2d complete:** `train_step` gradient descent for a single neuron, covered by 17 unit tests.
- **Step 2c complete:** `mean_squared_error` loss function, covered by 20 unit tests.
- **Step 2b complete:** `Network` class that chains layers, covered by 15 unit tests.
- Standardized the header format across all code files.

#### 2026-10-04
- **Documentation:** README, CONTRIBUTING, Code of Conduct, glossary, architecture, biology, evaluation, and ADR 0001.
- **Step 1 complete:** `Neuron` class with a numerically stable sigmoid, covered by 22 unit tests.
- **Step 2a complete:** `Layer` class, covered by 17 unit tests.
- **Policy:** CONTRIBUTING updated to permit AI-drafted tests.
- Default branch set to `master`.
