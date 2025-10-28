# Doc

This folder contains theoretical property analysis code, which is used to produce Section 2 of the LCO Embedding paper.

# Example Usage

To use these code with your own model, follow roughly the following step, which we provide using LCO-Embedding as example:


1. first use `encode.py` to get a `image_embeddings.pt` and `text_embeddings.pt`. Both `image_embeddings.pt` and `text_embeddings.pt` should have the size of (num_of_layers, num_of_examples, dimensionality). For example, if you encode 1000 images and texts with LCO-Embedding-Omni-7B, they will have the shape of (28, 1000, 3584). 

You will need to change the path in the `load_dataset` part of `encode.py`.

Analysis:

2. For anisotropy estimate, run
```
python anisotropy.py
```
remember to change the file name of the pre-encoded `image_embeddings.pt` and `text_embeddings.pt` to the path that you put it.

3. For kernel similarity structure, run
```
python kernel_alignment.py
```

also, change the file name of the pre-encoded `image_embeddings.pt` and `text_embeddings.pt` to the path that you put it.

Above we provide example for image-text analysis using PixmoCaps, you can easily extend it for video-text, audio-text, with msrvtt and audiocaps like we did in the paper, but here we omit them for brevity!


# Reference

Anisotropy estimate code adapted from:
```
@inproceedings{xiao2023isotropy,
  title={On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning},
  author={Xiao, Chenghao and Long, Yang and Al Moubayed, Noura},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={12266--12283},
  year={2023}
}
```

Kernel similarity structure analysis code adapted from:
```
@inproceedings{huh2024position,
  title={Position: The platonic representation hypothesis},
  author={Huh, Minyoung and Cheung, Brian and Wang, Tongzhou and Isola, Phillip},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```