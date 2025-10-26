from transformers import Qwen2_5OmniForConditionalGeneration
import torch
from peft import PeftModel

lora_path = "./checkpoint/LCO_Omni_test/checkpoint-704"
merge_path = "./checkpoint/LCO_Omni_test/checkpoint-704-merged"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B",
                                    device_map='auto',torch_dtype=torch.bfloat16)

model.thinker = PeftModel.from_pretrained(model.thinker, lora_path).merge_and_unload()
model.save_pretrained(merge_path)
