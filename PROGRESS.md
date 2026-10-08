## Progress

### Current Status
Working on **step 3: Habitat (grid world)**.

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
| **3** | Habitat (grid world) | ⏳ Next |

### Log
#### 2026-10-07
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
