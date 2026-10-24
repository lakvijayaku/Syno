## ADR 0002: Rewrite SYNO in Rust

### Status
Accepted

### Context
Phase 1 built SYNO in pure Python with only the standard library. Profiling a training run in the habituation experiment shows that `Neuron.forward` takes 55% of the time and backpropagation (`train_network_step`) 44%, so about 99% is neural-network arithmetic running in interpreted loops. Measuring every signature over 10 seeds takes about 23 minutes on a MacBook Air M4, which makes larger brains and worlds impractical. A partial rewrite is possible with only standard libraries, since Python's `ctypes` can call a Rust library, but the slow work is hundreds of thousands of tiny calls per run, such as one `Neuron.forward` per neuron per step, and each would cross the language boundary, so most of the loop would have to move to Rust anyway. Rewriting every file by hand also means every line of SYNO is understood again as it is ported.

### Decision
Phase 2 rewrites all of SYNO in Rust by hand, using only the Rust standard library, on the `rust_master` branch. Each commit replaces one Python file and its tests with a Rust file at the same path, such as `syno/brain/neuron.py` with `syno/brain/neuron.rs`, starting from the bottom of the dependency tree (`neuron`, `layer`, `network`, `training`). `Cargo.toml` points Cargo at `syno/` instead of the usual `src/`. `rust_master` is merged into `master` as v2.0.0 once the Rust SYNO reproduces the signatures.

### Consequences
- `master` keeps the working Python SYNO until the merge, so its tests stay green and v1.0.0 remains the version to run.
- On `rust_master`, the Python SYNO stops running once its first file is replaced, but every commit must build and pass `cargo test`, so the Rust side is never broken.
- The Python SYNO at v1.0.0 is the reference implementation. The Rust SYNO cannot match it bit for bit, because it uses its own random number generator, so it passes when every signature in [`evaluation.md`](../evaluation.md), measured over 10 seeds, reaches the same status as in the Python SYNO: reproduced, reproduced weakly, or not reproduced.
- Documentation fixes made on `master` during the rewrite must be merged into `rust_master`, so the two branches do not drift apart.
- The dependency rule carries over: no crates, and any dependency that becomes necessary is pinned, hash-verified, and vendored.
