from __future__ import annotations

from functools import partial
from typing import Any
import logging
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

tensor_to_image = transforms.Compose([transforms.ToPILImage()])

from PIL import Image

class Omni3Wrapper:
    def __init__(
        self,
        model_name: str,
        composed_prompt=None,
        **kwargs: Any,
    ):
    
        self.model_name = model_name
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name,max_pixels = 1280*28*28)
        # self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name) # for doc, such as Vidore, use full resolution
        self.processor.tokenizer.padding_side = "left"
        self.processor.padding = True
        if "device" in kwargs:
            self.device = kwargs.pop("device")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.text_prompt =  "{}\nSummarize the above text in one word:" # might exist better ones but default
        self.image_prompt =  "{}\nSummarize the above image in one word:" # might exist better ones but default

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]

                # due to OOM, first truncate texts manually before putting them in promtps.
                pre_tokenized_texts = self.processor.tokenizer(batch_texts).input_ids
                pre_tokenized_texts = [x[:300] for x in pre_tokenized_texts]
                batch_texts = [self.processor.tokenizer.decode(x) for x in pre_tokenized_texts]

                batch_texts = [self.text_prompt.format(text) for text in batch_texts]
                messages = [[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text":text},
                        ],

                    }
                ] for text in batch_texts]
                text_inputs = self.processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
                text_inputs = self.processor(
                text = text_inputs,
                padding = True,
                return_tensors = "pt",
                )
                text_inputs = text_inputs.to("cuda")
                text_outputs = self.model(
                    **text_inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                all_text_embeddings.append(text_outputs.to(torch.float16).cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            if isinstance(images, DataLoader):
                for batch_images in tqdm(images):
                    batch_images = [
                        tensor_to_image(image) for image in batch_images
                    ]
                    messages = [[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image":image},
                                {"type": "text", "text":"\nSummarize the above image in one word:"},
                            ],

                        }
                    ] for image in batch_images]
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=True)
                    inputs = self.processor(
                        text=text, 
                        audios=audio_inputs, 
                        images=image_inputs, 
                        videos=video_inputs, 
                        return_tensors="pt", 
                        padding=True
                    )
                    inputs = inputs.to("cuda")
                    image_outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    ).hidden_states[-1][:, -1, :]
                    all_image_embeddings.append(image_outputs.to(torch.float16).cpu())
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    messages = [[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image":image},
                                {"type": "text", "text":"\nSummarize the above image in one word:"},
                            ],

                        }
                    ] for image in batch_images]
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=True)
                    inputs = self.processor(
                        text=text, 
                        audio=audio_inputs, 
                        images=image_inputs, 
                        videos=video_inputs, 
                        return_tensors="pt", 
                        padding=True
                    )
                    inputs = inputs.to("cuda")
                    image_outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    ).hidden_states[-1][:, -1, :]
                    all_image_embeddings.append(image_outputs.to(torch.float16).cpu())

        return torch.cat(all_image_embeddings, dim=0)
    def get_video_embeddings(
        self,
        videos: list[str] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        long_video: bool = False,
        **kwargs: Any,
        ):
        all_video_embeddings = []
        with torch.no_grad():
            if isinstance(videos, DataLoader):
                ## TODO
                raise NotImplementedError
            else:
                for i in tqdm(range(0, len(videos), batch_size)):
                    torch.cuda.empty_cache()
                    
                    batch_videos = videos[i : i + batch_size]
                    if long_video:
                        messages = [[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video", 
                                        "video": video, 
                                        "max_pixels": 224 * 224,
                                        "fps": 1,
                                        "max_frames": 50
                                    },
                                    {"type": "text", "text":"\nSummarize the above video in one word:"},
                                ],

                            }
                        ] for video in batch_videos]
                    else:
                        messages = [[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video", 
                                        "video": video, 
                                    },
                                    {"type": "text", "text":"\nSummarize the above video in one word:"},
                                ],

                            }
                        ] for video in batch_videos]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    audio_inputs, image_inputs, video_inputs = process_mm_info(
                        messages, use_audio_in_video=False
                    )
                    

                    inputs = self.processor(
                        text=text, 
                        audio=audio_inputs, 
                        images=image_inputs, 
                        videos=video_inputs, 
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    
                    inputs = inputs.to("cuda")
                    video_outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    ).hidden_states[-1][:, -1, :]   
                    all_video_embeddings.append(video_outputs.to(torch.float16).cpu())
                    
                    del inputs, video_outputs
                    torch.cuda.empty_cache()
                        
        return torch.cat(all_video_embeddings, dim=0)
    def get_audio_embeddings(
        self,
        audios: list[str] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
        ):
        all_audio_embeddings = []
        sys_prompt='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
        with torch.no_grad():
            if isinstance(audios, DataLoader):
                ## TODO
                raise NotImplementedError
            else:
                for i in tqdm(range(0, len(audios), batch_size)):
                    torch.cuda.empty_cache()
                    
                    batch_audios = audios[i : i + batch_size]
                    messages = [[
                        # {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                        {
                            "role": "user",
                            "content": [
                                 {"type": "audio", "audio": audio},
                                {"type": "text", "text":"\nSummarize the above audio in one word:"},
                            ],
                            
                        }
                    ] for audio in batch_audios]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    audio_inputs, image_inputs, video_inputs = process_mm_info(
                        messages, use_audio_in_video=False
                    )
                    
                    inputs = self.processor(
                        text=text, 
                        audio=audio_inputs, 
                        images=image_inputs, 
                        videos=video_inputs, 
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    
                    inputs = inputs.to("cuda")
                    audio_outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    ).hidden_states[-1][:, -1, :]   
                    all_audio_embeddings.append(audio_outputs.to(torch.float16).cpu())
                    del inputs, audio_outputs
                    torch.cuda.empty_cache()
                        
        return torch.cat(all_audio_embeddings, dim=0)

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] = None,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        all_fused_embeddings = []
        kwargs.update(batch_size=batch_size)

        if texts is not None and images is not None:
            with torch.no_grad():
                if isinstance(images, DataLoader):
                    for index, batch_images in enumerate(tqdm(images)):
                        batch_images = [
                            tensor_to_image(image) for image in batch_images
                        ]
                        batch_texts = texts[
                            index * batch_size : (index + 1) * batch_size
                        ]

                        # due to OOM, first truncate texts manually before putting them in promtps.
                        pre_tokenized_texts = self.processor.tokenizer(batch_texts).input_ids
                        pre_tokenized_texts = [x[:300] for x in pre_tokenized_texts]
                        batch_texts = [self.processor.tokenizer.decode(x) for x in pre_tokenized_texts]
                        
                        assert len(batch_images) == len(batch_texts)
                        messages = [[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text":text},
                                    {"type": "image", "image":image},
                                    {"type": "text", "text":"\nSummarize the above in one word:"}
                                ]
                            }
                        ] for text,image in zip(batch_texts, batch_images)]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                        inputs = self.processor(
                            text=text, 
                            audio=audios, 
                            images=images, 
                            videos=videos, 
                            return_tensors="pt", 
                            padding=True
                        )
                        inputs = inputs.to("cuda")
                        image_outputs = self.model(
                            **inputs, output_hidden_states=True, return_dict=True
                        ).hidden_states[-1][:, -1, :]

                        all_fused_embeddings.append(image_outputs.to(torch.float16).cpu())
                else:
                    if len(texts) != len(images):
                        raise ValueError(
                            "The number of texts and images must have the same length"
                        )
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_texts = texts[i : i + batch_size]
                        batch_images = images[i : i + batch_size]
                        assert len(batch_images) == len(batch_texts)
                        messages = [[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text":text},
                                    {"type": "image", "image":image},
                                    {"type": "text", "text":"\nSummarize the above in one word:"},
                                ],

                            }
                        ] for text,image in zip(batch_texts, batch_images)]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=True)
                        inputs = self.processor(
                            text=text, 
                            audio=audio_inputs, 
                            images=image_inputs, 
                            videos=video_inputs, 
                            return_tensors="pt", 
                            padding=True
                        )
                        inputs = inputs.to("cuda")
                        image_outputs = self.model(
                            **inputs, output_hidden_states=True, return_dict=True
                        ).hidden_states[-1][:, -1, :]

                        all_fused_embeddings.append(image_outputs.to(torch.float16).cpu())
            return torch.cat(all_fused_embeddings, dim=0)
        elif texts is not None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)
            return text_embeddings
        elif images is not None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)
            return image_embeddings


omni_multimodal7b = ModelMeta(
    loader=partial(
        Omni3Wrapper,
        model_name="LCO-Embedding/LCO-Embedding-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ),
    name="LCO-Embedding-Omni-7B",
    languages=["eng_Latn"],
    revision="3d38f58aae1253a4443b1270b0767f1e533936cf",
    release_date="2025-03-27",
    modalities=["image", "text"],
    n_parameters=1, # fill in later
    memory_usage_mb=1, # fill in later
    max_tokens=1, # fill in later
    embed_dim=1, # fill in later
    license=None,
    open_weights=True,
    public_training_code="https://github.com/LCO-Embedding/LCO-Embedding/tree/main",
    public_training_data="https://huggingface.co/LCO-Embedding",
    framework=["PyTorch"],
    reference="https://arxiv.org/abs/2510.11693",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # LCO-Embedding training
    }
)

omni_multimodal3b = ModelMeta(
    loader=partial(
        Omni3Wrapper,
        model_name="LCO-Embedding/LCO-Embedding-Omni-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ),
    name="LCO-Embedding-Omni-3B",
    languages=["eng_Latn"],
    revision="eea763cfaf673e955ae86c64968896a3fea70189",
    release_date="2025-03-27",
    modalities=["image", "text"],
    n_parameters=1, # fill in later
    memory_usage_mb=1,# fill in later
    max_tokens=1, # fill in later
    embed_dim=1, # fill in later
    license=None,
    open_weights=True,
    public_training_code="https://github.com/LCO-Embedding/LCO-Embedding/tree/main",
    public_training_data="https://huggingface.co/LCO-Embedding",
    framework=["PyTorch"],
    reference="https://arxiv.org/abs/2510.11693",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # LCO-Embedding training
    }
)