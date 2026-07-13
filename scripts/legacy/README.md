# Legacy scripts

These files are the original RQ1/RQ2 one-off scripts and SLURM wrappers used
to produce the committed results.  They are preserved here for reproducibility.

For new runs, prefer:

```bash
python scripts/rq.py list
python scripts/rq.py <command> [args...]
```

The top-level [`scripts/rq.py`](../rq.py) dispatcher calls these legacy scripts
under the hood, so moving them here should not change the experiment logic.
