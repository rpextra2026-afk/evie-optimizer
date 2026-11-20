# Evie CPU checks

synthetic CPU checks for the finite-horizon analysis and its correspondence with the implemented Evie and Evie-Amplify optimizers. the suite uses float64 arithmetic, fixed seeds, and local source files; it does not launch the finance, language, or vision training experiments.

## setup

from the repository root:

```bash
python -m pip install -r cpu_checks/requirements.txt
PYTHONPATH=cpu_checks python -m unittest discover -s cpu_checks/tests
```

## running

```bash
python cpu_checks/run.py --profile smoke --source-root .
python cpu_checks/run.py --profile core --source-root . --workers 4
python cpu_checks/run.py --profile core --source-root . --resume RUN_ID
python cpu_checks/run.py --verify RUN_ID
```

use `--output-root /path/to/local-output` to choose where generated artifacts are written. the default is `cpu_checks/results/`, which is gitignored. use the same output root when resuming or verifying a run.

the implementation uses one worker and one BLAS/PyTorch thread; `--workers` is a maximum. all tensors are on CPU. completed groups can be resumed only when configuration, optimizer-source, and engine hashes match. the optional extended profile is explicitly rejected because its additional cases are not implemented.

## check groups

| group | purpose |
| --- | --- |
| R0 | agreement between independent exact-loss calculations |
| R1 | correspondence between optimizer implementations and the independent functional calculation |
| R2 | calibrated gate comparisons after independent learning-rate tuning |
| R3 | finite-horizon and momentum expansions |
| R4 | noncommuting matrix checks |
| R5 | adaptive-reference, calibration, and transfer checks |
| R6 | local-advantage bounds and supporting scalar checks |

costs include step zero and exclude the terminal step. learning-rate tuning applies the same stability rules, prescribed grids, and independent validation grid to each shape. the online tuning and evaluation seed sets are separate, with coupled noise within each comparison. the reference is `ungated_matched`.

## optimizer sources

the two standalone reference modules are bundled in `reference_optimizers/`. the remaining six classes are read from the three existing `benchmark_partA.py` files using AST extraction, so importing a benchmark does not start training. their update logic matches the source used by the CPU suite, with the public Evie class names retained. the existing training code and numerical grids are unchanged.

generated tables, figures, traces, checkpoint files, and numerical reports stay in the selected output folder. the runner records numerical failures and unfinished obligations rather than treating numerical agreement as a proof. saved CPU run outputs are not part of this source release.
