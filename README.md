<div align="center">

<h1>Scaling Language-centric Omnimodal Representation Learning</h1>

<div>
    <a target='_blank'>Chenghao Xiao,</a>&emsp;
    <a target='_blank'>Hou Pong Chan<sup>†</sup>,</a>&emsp;
    <a target='_blank'>Hao Zhang<sup>†</sup>,</a>&emsp;
    <a target='_blank'>Weiwen Xu,</a>&emsp;
    <a target='_blank'>Mahani Aljunied,</a>&emsp;
    <a target='_blank'>Yu Rong<sup>‡</sup></a>&emsp;
</div>

<div>
    <em>DAMO Academy, Alibaba Group</em>&emsp;
</div>
<em><sup>†</sup>Corresponding Authors &emsp;<sup>‡</sup>Project Head</em>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.11693-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2510.11693)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2510.11693)
[![hf_collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collections-blue.svg)](https://huggingface.co/MMR1/datasets)
<br>
</h5> 


</div>

<h5 align="center"> 🌟 This repo contains the codes and datasets for the paper "Scaling Language-centric Omnimodal Representation Learning" to appear at NeurIPS 2025. If our project helps you, please give us a star ⭐ on GitHub and upvote our HF paper to support us. 🙏🙏 </h2>


<h2>🎉 Updates</h2>

- **[2025-10]** Check out our [paper](https://huggingface.co/papers/2510.11693) on Huggingface Daily Papers.
- **[2025-09]** Our paper is accepted by NeurIPS 2025.

<h2><img src="./assets/LCO-logo.png" width="30"> Overview</h2>

- We introduce **LCO-Embedding**, a language-centric omnimodal representation learning method and the LCO-Embedding model families, setting a new state-of-the-art on [MIEB](https://huggingface.co/blog/isaacchung/introducing-mieb) (Massive Image Embedding Benchmark), while supporting audio and videos.
- We introduce the **Generation-Representation Scaling Law**, and connect models' generative capabilities and their representation upper bound.
- We introduce **SeaDoc**, a challenging visual document retrieval task in Southeast Asian languages, and show that continual generative pretraining before contrastive learning raises the representation upper bound.

<div align='center'><img src="https://cdn-uploads.huggingface.co/production/uploads/604f67ef0fe8ff3ec13d71ef/4Wd8fDFBdT6GxqN6-KzZN.png" alt="overview" width="100%"/></div>


<h2>📊 Evaluation Results</h2>

We evaluate LCO-Embedding with the state-of-the-art embedding models, including E5-V, Voyage Multimodal 3, mmE5, and GME, on a MIEB-Lite benchmark (51 tasks) broken down by task categories.  

<div align='center'><img src="https://cdn-uploads.huggingface.co/production/uploads/63108cc834c7d77420b0fd68/63WBsKh57HbNwwe3bZ-oZ.png" alt="mieb_lite" width="100%"/></div>

Performance and efficiency comparisons of different training strategies using 3B and 7B variants of Qwen2.5-VL backbones.

<div align='center'><img src="./assets/lora_ablation.png" alt="mieb_lite" width="100%"/></div>

Scaling relationship between generation benchmark performance (X-axis) and representation benchmark performance after language-centric contrastive learning (Y-axis).

<div align='center'><img src="./assets/scaling.png" alt="mieb_lite" width="100%"/></div>



<h2>🔧 Getting Started</h2>

Ongoing: We are updating all code and resources.


<h2>📑 Citation</h2>

If you find LCO-Embedding useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{xiao2025scaling,
    title={Scaling Language-Centric Omnimodal Representation Learning}, 
    author={Chenghao Xiao and Hou Pong Chan and Hao Zhang and Weiwen Xu and Mahani Aljunied and Yu Rong},
    year={2025},
    eprint={2510.11693},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2510.11693}, 
}
```
