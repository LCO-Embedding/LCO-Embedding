from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

lora_path = "./checkpoint/LCO_VL_test/checkpoint-704"
merge_path = "./checkpoint/LCO_VL_test/checkpoint-704-merged"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                    device_map='auto',torch_dtype="auto")

model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
model.save_pretrained(merge_path)