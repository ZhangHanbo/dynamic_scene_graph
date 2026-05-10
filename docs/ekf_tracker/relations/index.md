# `ekf_tracker/relations/`

Scene-graph edges (`on` / `in` / `under` / `contain`).
`relation_orchestrator.py` decides _when_ to re-evaluate edges (grasp /
release / new object / every $N$ frames) and dispatches to one of the
backends in `relation_client.py` (REST or LLM). `relation_filter.py`
smooths the per-edge score with an EMA before emit, and `relation_utils.py`
provides the held-set transitive closure used by the manipulation
pipeline.

```{toctree}
:maxdepth: 1

relation_orchestrator
relation_filter
relation_client
relation_utils
```
