from __future__ import annotations

from functools import partial
from typing import Any
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
# Updated imports for the new model
from transformers import MllamaForConditionalGeneration, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

tensor_to_image = transforms.Compose([transforms.ToPILImage()])

class mmE5Wrapper:
    """
    Model wrapper for intfloat/mmE5-mllama-11b-instruct.
    """
    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        # For feature extraction with last-token pooling, right padding is preferred.
        self.processor.tokenizer.padding_side = "right"
        
        self.device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Load the Mllama model
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name, **kwargs
        ).to(self.device)
        self.model.eval()

        # Define prompts based on mmE5's format
        self.text_prompt = "Represent this sentence for searching relevant passages: {}"
        self.image_prompt = "<|image|><|begin_of_text|>Represent the given image.\n"
        self.fused_prompt = "<|image|><|begin_of_text|>Represent the given image with the following question: {}\n"
        # self.text_prompt = "Represent the text: {}"
        # self.image_prompt = "<|image|><|begin_of_text|>Represent the image.\n"
        # self.fused_prompt = "<|image|><|begin_of_text|>Represent the image and the text: {}\n"

    @staticmethod
    def _last_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Finds the last non-padding token's hidden state and normalizes it.
        """
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        return torch.nn.functional.normalize(reps, p=2, dim=-1)

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts"):
                batch_texts = texts[i : i + batch_size]
                
                # Manual truncation logic from your original code
                tokenized = self.processor.tokenizer(batch_texts).input_ids
                truncated_ids = [ids[:8000] for ids in tokenized]
                batch_texts = [self.processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in truncated_ids]
                
                prompts = [self.text_prompt.format(text) for text in batch_texts]
                
                inputs = self.processor(
                    text=prompts,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                ).to(self.device)

                hidden_states = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1]
                
                embeddings = self._last_pooling(hidden_states, inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        prompt = self.image_prompt

        with torch.no_grad():
            if isinstance(images, DataLoader):
                for batch_images in tqdm(images, desc="Encoding Images"):
                    batch_images = [tensor_to_image(image) for image in batch_images]
                    
                    inputs = self.processor(
                        text=[prompt] * len(batch_images),
                        # -> FIX: Wrap each image in its own list
                        images=[[img] for img in batch_images],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    hidden_states = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                    embeddings = self._last_pooling(hidden_states, inputs['attention_mask'])
                    all_embeddings.append(embeddings.cpu())
            else:
                for i in tqdm(range(0, len(images), batch_size), desc="Encoding Images"):
                    batch_images = images[i : i + batch_size]
                    
                    inputs = self.processor(
                        text=[prompt] * len(batch_images),
                        # -> FIX: Wrap each image in its own list
                        images=[[img] for img in batch_images],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)

                    hidden_states = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                    embeddings = self._last_pooling(hidden_states, inputs['attention_mask'])
                    all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        # ... (initial checks are the same) ...
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided.")
        if texts is not None and images is None:
            return self.get_text_embeddings(texts, batch_size=batch_size, **kwargs)
        if images is not None and texts is None:
            return self.get_image_embeddings(images, batch_size=batch_size, **kwargs)
        # if len(texts) != len(images):
        #     raise ValueError("The number of texts and images must be the same length.")

        all_embeddings = []
        with torch.no_grad():
            image_iterator = images if not isinstance(images, DataLoader) else iter(images)
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Fused"):
                batch_texts = texts[i : i + batch_size]
                
                if isinstance(images, DataLoader):
                    batch_images = next(image_iterator)
                    batch_images = [tensor_to_image(img) for img in batch_images]
                else:
                    batch_images = images[i : i + batch_size]

                # ... (text processing is the same) ...
                tokenized = self.processor.tokenizer(batch_texts).input_ids
                truncated_ids = [ids[:8000] for ids in tokenized]
                batch_texts = [self.processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in truncated_ids]
                prompts = [self.fused_prompt.format(text) for text in batch_texts]

                inputs = self.processor(
                    text=prompts,
                    # -> FIX: Wrap each image in its own list
                    images=[[img] for img in batch_images],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                hidden_states = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                embeddings = self._last_pooling(hidden_states, inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)

    def calculate_probs(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates similarity probabilities. Assumes embeddings are pre-normalized.
        """
        # Normalization is now handled by the _last_pooling function
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

mme5 = ModelMeta(
    loader=partial(
        mmE5Wrapper,
        model_name="/mnt/workspace/workgroup/chx/models/mmE5-mllama-11b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    ),
    name="mme5",
    languages=["eng_Latn"],
    revision="gowitheflow1998_2025_0408",
    release_date="2024-07-17",
    modalities=["image", "text"],
    n_parameters=8_360_000_000,
    memory_usage_mb=15936,
    max_tokens=8192,
    embed_dim=4096,
    license=None,
    open_weights=True,
    public_training_code="hhttps://huggingface.co/gowitheflow",
    public_training_data="https://huggingface.co/gowitheflow",
    framework=["PyTorch"],
    reference="https://huggingface.co/gowitheflow",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # princeton-nlp/datasets-for-simcse
    },
)
