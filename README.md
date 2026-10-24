## SYNO

[![tests](https://img.shields.io/github/actions/workflow/status/lakvijayaku/Syno/tests.yml?branch=master&label=tests)](https://github.com/lakvijayaku/Syno/actions/workflows/tests.yml) [![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

**Synthetic Yearning Neural Organism**

### Description
SYNO is a brain-inspired artificial organism built from scratch as a learning project. Instead of being trained purely on a fixed objective, SYNO is driven by internal motivations: artificial dopamine, drives such as hunger, thirst, and novelty, memory of past experiences, a short working memory, neuron growth, and hormones that act as slow, global modulators of learning, loosely modeled on human motivation.

### Project Status
Phase 1 is in progress: its 19 roadmap steps are complete. SYNO lives in a grid world, has a body with energy, a stomach, water, fatigue, and injury, and learns from its own dopamine signal, with reward coming only from reducing hunger, thirst, tiredness, and pain. It explores out of curiosity, replays its most surprising memories to learn faster, even when it replays only during sleep, grows new neurons when it stops learning, remembers what it saw for a few steps when its eyes are closed, adapts faster to a changing world through a surprise-driven arousal hormone, and in experiments it learns to keep itself fed and watered without being told how, dying if either runs out, to sleep when it is tired, to live among thorns that hurt it, and to leave places it has just explored and return to them later. Every component is covered by unit tests. SYNO's behavior has been measured against fourteen biological signatures: eleven are reproduced, one weakly, and two are not. See [`docs/evaluation.md`](docs/evaluation.md), and [`docs/humanness.md`](docs/humanness.md) for how many of 104 human capabilities SYNO has, which scores it at 29.5%. See [`PROGRESS.md`](PROGRESS.md) for details.

### Running SYNO
SYNO needs only Python 3.10 or later. Run the tests with `python3 -m unittest`, and an experiment with `python3 -m experiments.<name>`, for example `python3 -m experiments.consolidation`. Experiments save their data and charts to `runs/`. Slow tests that re-run whole experiments are skipped unless `SYNO_SLOW_TESTS=1` is set. On GitHub, the fast tests run automatically on every push, on Python 3.10 and 3.13.

### Architecture
SYNO consists of a simulated world, the Habitat, and an Artificial Mind (AM) that perceives, decides, and learns within it. Only physiology is hard-coded; all behavior must emerge from learning. See [`docs/architecture.md`](docs/architecture.md) for the full design, [`docs/biology.md`](docs/biology.md) for its biological basis, and [`docs/glossary.md`](docs/glossary.md) for component names.

### AI-Assisted Development
SYNO does not accept "vibe-coded" contributions. All project code must be written by a human, and every commit must be made by a human. AI tools may draft tests and documentation, which the contributor must verify. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full policy, and [`CLAUDE.md`](CLAUDE.md) for the rules Claude Code follows when tutoring on SYNO.

### Contact
Questions, bug reports, and suggestions can be sent to the project owner at [`lakvijayaku@gmail.com`](mailto:lakvijayaku@gmail.com).

### License
SYNO is a free and open-source project, licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for details.
