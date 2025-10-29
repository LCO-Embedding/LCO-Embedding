## LCO-Embedding Training resources

We will continue to update resources and guidelines to cook state-of-the-art omnimodal representation models and hope these will contribute the community!

## Env

```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## VL training scripts

Prepare a jsonl file where each dict is a:
```
{
"anchor":{"image_path": "",
"text": "What is LCO-Embedding?"
},
"positive":{"image_path": "",
"text": "LCO Embedding is an omnimodal representation model family that's trained using language-centric paradigm."
},
"negative":{"image_path": "",
"text": "LOCO is a Korean Hip Hop artist."
}
}
```

Either anchor, positive and negative can have image, text, or both (it's `any2any`!). If one only has text, put an empty string "" for `image_path`, if only image, put an empty string "" for `text`.
The `image_path` can be an absolute path (e.g., `/chenghao/LCO_Embedding/data/image_data0001/a_random_dog_pic101.jpg`) or a path relative to the training script (e.g., `data/image_data0001/a_ramdom_cat_pic101.jpg`).


put it under `data/your_jsonl_file_name.jsonl` then run
```
bash train_VL.sh
```

If you're training using many GPUs, e.g., 128 GPUs with 16 nodes, we recommend changing deepspeed config in `train_VL.sh` to `deepspeed_config/ds_manynodes.config` which stores states per node which otherwise would crash the main node in our experience.

Enjoy!

## Omni training scripts & resources

Qwen2.5-Omni can be run with:
```
bash train_omni.sh
```

The omni training code currently only supports image, text and image-text interleaved as VL code above, aligning with what we did in the paper. We will support all modalities (audio, video) after we test the optimal setting soon.

## merge LoRA weights

After training, run `merge_Omni.py` or `merge_VL.py` to merge the LoRA weights to the base model, and move other non-checkpoint files from the base model folder to the merged model folder, including `chat_template.json`, `merges.txt`, `preprecessor_config.json`, `tokenizer_config.json`, `tokenizer.json`, `vocab.json` for `Qwen2.5-VL` models, plus an extra `spk_dict.pt` for `Qwen2.5-Omni` models.
