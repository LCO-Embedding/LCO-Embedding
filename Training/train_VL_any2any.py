import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import fire
import torch
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import set_seed, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

@dataclass
class DataCollatorForAny2AnyContrastive:
    """
    Data collator that handles dynamic batching of image-text data.
    It takes a list of dataset examples and prepares the final model-ready tensors
    for contrastive learning with anchors, positives, and hard negatives.
    """
    processor: AutoProcessor
    max_length: int
    
    def _process_item(self, item_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
        image_prompt = "\nSummarize the above image in one word:"
        text_prompt = "\nSummarize the above text in one word:"
        interleaved_prompt = "\nSummarize the above in one word:"

        content = []

        if item_data["image_path"]:
            has_image = True
        else:
            has_image = False
        if item_data["text"]:
            has_text = True
        else:
            has_text = False
        
        if has_image:
            image_path = item_data["image_path"]
            pil_image = Image.open(image_path).convert("RGB")
            pil_image.thumbnail((700, 700))
            content.append({"type": "image", "image": pil_image})
        
        if has_text:
            text = item_data["text"]
            pre_tokenized_text = self.processor.tokenizer(text).input_ids
            pre_tokenized_text = pre_tokenized_text[:self.max_length]
            text = self.processor.tokenizer.decode(pre_tokenized_text)
            content.append({"type": "text", "text": text})
        if has_image and has_text:
            content.append({"type": "text", "text": interleaved_prompt})
        elif has_image:
            content.append({"type": "text", "text": image_prompt})
        elif has_text:
            content.append({"type": "text", "text": text_prompt})
        message = [{"role": "user", "content": content}]
        return message

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        batch_anchor_messages = []
        batch_positive_messages = []
        batch_negative_messages = []

        for feature in features:
            anchor_message = self._process_item(feature["anchor"])
            batch_anchor_messages.append(anchor_message)

            positive_message = self._process_item(feature["positive"])
            batch_positive_messages.append(positive_message)

            negative_message = self._process_item(feature["negative"])
            batch_negative_messages.append(negative_message)


        batch_anchor_prompts = self.processor.apply_chat_template(batch_anchor_messages, tokenize=False, add_generation_prompt=True)
        batch_positive_prompts = self.processor.apply_chat_template(batch_positive_messages, tokenize=False, add_generation_prompt=True)
        batch_negative_prompts = self.processor.apply_chat_template(batch_negative_messages, tokenize=False, add_generation_prompt=True)

        loaded_anchor_images, _ = process_vision_info(batch_anchor_messages)
        loaded_positive_images, _ = process_vision_info(batch_positive_messages)
        loaded_negative_images, _ = process_vision_info(batch_negative_messages)

        anchor_inputs = self.processor(
            text=batch_anchor_prompts,
            images=loaded_anchor_images,
            return_tensors="pt",
            padding=True
        )
        
        positive_inputs = self.processor(
            text=batch_positive_prompts,
            images=loaded_positive_images,
            return_tensors="pt",
            padding=True
        )
        
        negative_inputs = self.processor(
            text=batch_negative_prompts,
            images=loaded_negative_images,
            return_tensors="pt",
            padding=True
        )

        final_batch = {
            "anchor_input_ids": anchor_inputs["input_ids"],
            "anchor_attention_mask": anchor_inputs["attention_mask"],
            "positive_input_ids": positive_inputs["input_ids"],
            "positive_attention_mask": positive_inputs["attention_mask"],
            "negative_input_ids": negative_inputs["input_ids"],
            "negative_attention_mask": negative_inputs["attention_mask"],
        }

        for prefix, inputs in [("anchor", anchor_inputs), ("positive", positive_inputs), ("negative", negative_inputs)]:
            if "pixel_values" in inputs:
                final_batch[f"{prefix}_pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                final_batch[f"{prefix}_image_grid_thw"] = inputs["image_grid_thw"]
  
        return final_batch


class ContrastiveTrainer(Trainer):

    def _get_embeddings(self, model, inputs: Dict[str, Any], prefix: str) -> Tuple[torch.Tensor, CausalLMOutputWithPast]:

        model_inputs = {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs[f"{prefix}_attention_mask"],
            "output_hidden_states": True,
            }
        if f"{prefix}_pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs[f"{prefix}_pixel_values"]
            model_inputs["image_grid_thw"] = inputs[f"{prefix}_image_grid_thw"]
        outputs = model(**model_inputs)
        embeds = outputs.hidden_states[-1][:, -1, :]
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds, outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes a one-way contrastive loss using anchors, positives, and hard negatives.
        """
        anchor_embeds, anchor_outputs = self._get_embeddings(model, inputs, "anchor")
        positive_embeds, positive_outputs = self._get_embeddings(model, inputs, "positive")
        negative_embeds, negative_outputs = self._get_embeddings(model, inputs, "negative")

        if dist.is_initialized():
            anchor_embeds_list = [torch.zeros_like(anchor_embeds) for _ in range(dist.get_world_size())]
            positive_embeds_list = [torch.zeros_like(positive_embeds) for _ in range(dist.get_world_size())]
            negative_embeds_list = [torch.zeros_like(negative_embeds) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=anchor_embeds_list, tensor=anchor_embeds.contiguous())
            dist.all_gather(tensor_list=positive_embeds_list, tensor=positive_embeds.contiguous())
            dist.all_gather(tensor_list=negative_embeds_list, tensor=negative_embeds.contiguous())

            anchor_embeds_list[dist.get_rank()] = anchor_embeds
            positive_embeds_list[dist.get_rank()] = positive_embeds
            negative_embeds_list[dist.get_rank()] = negative_embeds
            
            anchor_embeds = torch.cat(anchor_embeds_list, 0)
            positive_embeds = torch.cat(positive_embeds_list, 0)
            negative_embeds = torch.cat(negative_embeds_list, 0)


        all_candidates = torch.cat([positive_embeds, negative_embeds], dim=0)
        
        logits = torch.matmul(anchor_embeds, all_candidates.t()) / 0.03

        batch_size = anchor_embeds.size(0)
        labels = torch.arange(batch_size, device=model.device)

        loss = F.cross_entropy(logits, labels)
        
        return (loss, (anchor_outputs, positive_outputs, negative_outputs)) if return_outputs else loss


def train(
    base_model: str = "",
    data_path: str = "data/LCO_multimodal.jsonl",
    output_dir: str = "./LCO_multimodal",
    batch_size: int = 1056,
    per_device_batch_size: int = 33,
    num_epochs: int = 2,
    learning_rate: float = 3e-4,
    max_length: int = 650, 
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    save_steps: int = 100,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    grad_checkpoint: bool = False,
    bf16: bool = True,
    ):

    gradient_accumulation_steps = batch_size // per_device_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        dist.init_process_group("nccl")

    set_seed(seed)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
        torch_dtype=torch.bfloat16,
        device_map=device_map if ddp else "auto",
        trust_remote_code=True,
        attn_implementation=None,
    )

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    print("Freezing vision tower and visual projector...")
    for name, param in model.named_parameters():
        if 'vision_tower' in name or 'visual' in name:
            param.requires_grad = False

    if grad_checkpoint:
        model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)

    def find_all_llm_linear_names(model):
        """
        Finds the full path for all linear layer names in the language model part of 
        the Qwen-VL model. This prevents name collisions with the vision tower.
        """
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.model.named_modules():
            if isinstance(module, cls):
                lora_module_names.add(f"model.{name}")
        lora_module_names = {name for name in lora_module_names if 'lm_head' not in name}
        return list(lora_module_names)
    target_modules = find_all_llm_linear_names(model)
    print(f"Found LLM target modules for LoRA: {target_modules}")
        
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    data = load_dataset("json", data_files=data_path, split="train")
    train_data = data

    data_collator = DataCollatorForAny2AnyContrastive(
        processor=processor,
        max_length=max_length
    )

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            logging_steps=logging_steps,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=20,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            run_name=output_dir,
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
            remove_unused_columns=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none"
        ),
        data_collator=data_collator
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)