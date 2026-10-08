## Progress

### Current Status
Working on **step 5: Drives**.

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
| **5** | Drives | ⏳ Next |

### Log
#### 2026-10-07
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
