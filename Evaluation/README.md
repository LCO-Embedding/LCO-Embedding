## Usage

Here, I provide a snapshot of MIEB codebase (in around Feb 2025) when I developed it. This version is stable and will not be updated (for reproducibility for this project), and thus it might be a slightly different to the MIEB on MTEB overtime, which is constantly refactored and updated for newer functionality.

This codebase can support all MIEB evaluation (130 tasks); and all MTEB evaluation which was added to MTEB main branch before Feb 2025.

```
cd MTEB-MIEB
pip install -e .
```

example evaluation:

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

note that to use your own custom model, such as `LCO-Embedding-Omni-7B`, you'll need to implement and register your model under the `MTEB-MIEB/mteb/models` folder, see our `LCO_Embedding.py` for an example.


## Reference
```
@inproceedings{xiao2025mieb,
  title={Mieb: Massive image embedding benchmark},
  author={Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M{\'a}rton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22187--22198},
  year={2025}
}
```