## SYNO

**Synthetic Yearning Neural Organism**

### Description
SYNO is a brain-inspired artificial organism built from scratch as a learning project. Instead of being trained purely on a fixed objective, SYNO is driven by internal motivations: artificial dopamine, drives such as hunger and novelty, memory, neuron growth, and hormones that act as slow, global modulators of learning, loosely modeled on human motivation.

### Project Status
Phase 1 is in progress: 12 roadmap steps are complete, and step 13 gives SYNO thirst. SYNO lives in a grid world, has a body with energy and a stomach, and learns from its own dopamine signal, with reward coming only from reducing hunger. It explores out of curiosity, replays its most surprising memories to learn faster, grows new neurons when it stops learning, adapts faster to a changing world through a surprise-driven arousal hormone, and in experiments it learns to keep itself fed without being told how. Every component is covered by unit tests. SYNO's behavior has been measured against six biological signatures: two are reproduced, one weakly, two are not, and one cannot be measured in the current design. See [`docs/evaluation.md`](docs/evaluation.md), and [`docs/humanness.md`](docs/humanness.md) for how many of 104 human capabilities SYNO has, which scores it at 24%. See [`PROGRESS.md`](PROGRESS.md) for details.

### Running SYNO
SYNO needs only Python 3.10 or later. Run the tests with `python3 -m unittest`, and an experiment with `python3 -m experiments.<name>`, for example `python3 -m experiments.consolidation`. Experiments save their data and charts to `runs/`. Slow tests that re-run whole experiments are skipped unless `SYNO_SLOW_TESTS=1` is set.

### Architecture
SYNO consists of a simulated world, the Habitat, and an Artificial Mind (AM) that perceives, decides, and learns within it. Only physiology is hard-coded; all behavior must emerge from learning. See [`docs/architecture.md`](docs/architecture.md) for the full design, [`docs/biology.md`](docs/biology.md) for its biological basis, and [`docs/glossary.md`](docs/glossary.md) for component names.

### AI-Assisted Development
SYNO does not accept "vibe-coded" contributions. All project code must be written by a human, and every commit must be made by a human. AI tools may draft tests and documentation, which the contributor must verify. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full policy.

### Contact
Questions, bug reports, and suggestions can be sent to the project owner at [`lakvijayaku@gmail.com`](mailto:lakvijayaku@gmail.com).

### License
SYNO is a free and open-source project, licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for details.
