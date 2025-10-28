## MTEB-MIEB

a snapshot of MTEB/MIEB codebase (in around Feb 2025) for evaluation of LCO-Embedding project.

Run our example script:

```
python run-mieb-lite.py
```

Example code for run a few Visual STS tasks:

```python
import mteb
tasks=mteb.get_tasks(
    tasks=[
    "STSBenchmarkMultilingualVisualSTS",
    "STS17MultilingualVisualSTS"
    ]
)

for model_name in [
    "LCO-Embedding-Omni-7B",
    "LCO-Embedding-Omni-3B"
]:

    model = mteb.get_model(model_name)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="mieb-results", batch_size=8)
```