# import soundfile as sf
from tqdm import tqdm
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader
from torchvision import transforms
import math
import os
import os
import logging
from datasets import load_dataset
from PIL import Image

model_name = "LCO-Embedding/LCO-Embedding-Omni-7B"
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

unified_prompt =  "{}\nSummarize the above in one word:"
text_prompt =  "{}\nSummarize the above text in one word:" 
image_prompt =  "{}\nSummarize the above image in one word:"

def encoding_text_all_layers(model, 
                            processor, 
                            texts, 
                            batch_size = 16):

    all_text_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            batch_texts = [text_prompt.format(text) for text in batch_texts]
            messages = [[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":text},
                    ],

                }
            ] for text in batch_texts]
            text_inputs = processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
            text_inputs = processor(
            text = text_inputs,
            padding = True,
            return_tensors = "pt",
            )
            text_inputs = text_inputs.to("cuda")
            text_outputs = model(
                **text_inputs, output_hidden_states=True, return_dict=True
            ).hidden_states # num_layers * (num_examples, num_tokens, dimensionality)
            
            # to-do
            # if last-token pooling; if mean pooling...

            text_outputs = [text_output[:, -1, :] for text_output in text_outputs] 
            # num_layers * (num_examples, dimensionality)

            stacked_text_outputs = torch.stack(text_outputs)
            # (num_layers, num_examples, dimensionality)

            all_text_embeddings.append(stacked_text_outputs.cpu())
    all_text_embeddings = torch.cat(all_text_embeddings, dim=1) # cat over num_examples dim.

    return all_text_embeddings

def encoding_image_all_layers(model, 
                            processor, 
                            images, 
                            batch_size = 16):

    tensor_to_image = transforms.Compose([transforms.ToPILImage()])
    all_image_embeddings = []

    with torch.no_grad():
        if isinstance(images, DataLoader):
            for batch_images in tqdm(images):
                batch_images = [
                    tensor_to_image(image) for image in batch_images
                ]
                batch_images = [downsample_image(image) for image in batch_images]
                messages = [[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image":image},
                            {"type": "text", "text":"\nSummarize the above in one word:"},
                        ],

                    }
                ] for image in batch_images]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=True)
                inputs = processor(
                    text=text, 
                    audios=audio_inputs, 
                    images=image_inputs, 
                    videos=video_inputs, 
                    return_tensors="pt", 
                    padding=True
                )
                
                inputs = inputs.to("cuda")
                image_outputs = model.thinker(
                    **inputs, output_hidden_states=True, return_dict=True
                ).hidden_states

                image_outputs = [image_output[:, -1, :] for image_output in image_outputs] 
                # num_layers * (num_examples, dimensionality)

                stacked_image_outputs = torch.stack(image_outputs)
                # (num_layers, num_examples, dimensionality)
                all_image_embeddings.append(stacked_image_outputs.cpu())
        else:
            raise NotImplementedError
    all_image_embeddings = torch.cat(all_image_embeddings, dim=1)
    return all_image_embeddings


def downsample_image(
    image: Image.Image, max_pixels: int = 262144, target_longest_side: int = 512
) -> Image.Image:
    """If image pixel > max_pixels, downsample it to target_longest_side while keeping the width height ratio."""
    width, height = image.size
    pixels = width * height

    if pixels > max_pixels:
        if width > height:
            new_width = target_longest_side
            new_height = int(height * (target_longest_side / width))
        else:
            new_height = target_longest_side
            new_width = int(width * (target_longest_side / height))

        new_size = (new_width, new_height)
        logging.info(
            f"Downsampling image from {width}x{height} to {new_width}x{new_height}"
        )
        return image.resize(new_size, Image.LANCZOS)
    if width > height:
        if width > 10000:
            logging.error("Processing extremely wide images.")
            return image.resize((10000, height), Image.LANCZOS)
    else:
        if height > 10000:
            logging.error("Processing extremely high images.")
            return image.resize((width, 10000), Image.LANCZOS)
    return image


# we first download the dataset to local, then use a parquet such that it is easier to load.
# feel free to use a different dataset

dataset = load_dataset('parquet', data_files="/chenghao/embedding_data/pixmo-cap-images/data/train-00000-of-00075.parquet")
dataset = dataset["train"]
# dataset = dataset.shuffle()
dataset = dataset.select(range(1000))


batch_size = 8
def custom_collate_fn(batch):
    return batch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_column_name = image_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column_name]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            # Assume the image is already in a usable format (e.g., PIL Image)
            image = image
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
    


DEFAULT_TRANSFORM = transforms.Compose([transforms.PILToTensor()])

image_dataset = ImageDataset(
    dataset, image_column_name="image", transform=DEFAULT_TRANSFORM
)
image_dataloader = DataLoader(
    image_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=min(math.floor(os.cpu_count() / 2), 16),
)

texts = dataset["caption"]

image_embeddings = encoding_image_all_layers(model, 
                            processor, 
                            image_dataloader, 
                            batch_size = batch_size)

text_embeddings = encoding_text_all_layers(model, 
                            processor, 
                            texts, 
                            batch_size = batch_size)

torch.save(image_embeddings, './embeddings/image_embeddings_pixmo_omni.pt')
torch.save(text_embeddings, './embeddings/text_embeddings_pixmo_omni.pt')