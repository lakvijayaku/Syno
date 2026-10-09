## Progress

### Current Status
Steps 1 to 9 are complete. Step 10 (Evaluation signatures) is in progress: E2, E4, and E5 are measured, and E6 is not measurable with the current design.

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
| **10** | Evaluation signatures | 🔄 In progress |

### Log
#### 2026-10-08
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
