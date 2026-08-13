# Benchmarking scripts (historical)

These per-target scripts (`martinez/`, `metallohydrolases/`, `PBP_binding/`,
`PDE_2H/`, `serine_hydrolases/`, `yang_filterzyme/`, `yang_paper/`) are the
original benchmark runs used during development. They contain **dead,
machine-specific paths** (e.g. `/nvme2/helen/...`, old `Filterzyme` paths,
`conda activate filterzyme`) and are **not** guaranteed to run out of the box.

To reuse one: edit the hardcoded input/output paths and env name for your
environment, and prefer the current CLI (`structurezyme run --config ...`) over
the legacy `Pipeline` API where possible.
