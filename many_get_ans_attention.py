import os
from typing import Optional
import torch
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
)
from mmsg.utils import load_image
from prompts import get_ask_prompt
from dataset import get_question_answer_dataset
from tqdm import tqdm
import json
from mmsg.integrations.chameleon_utils import postprocess_token_sequence
import logging
import numpy as np
import matplotlib.pyplot as plt
from peft import PeftModel

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

def run_interleaved_generation(
    data_id,
    prompt: Optional[str] = None,
    image_paths: list = [],
    max_new_tokens: int = 3000,
    outputs_dir: str = "./outputs",
    attention_dir: str = "attentions",
):
    images = [load_image(image_path) for image_path in image_paths]
    inputs = processor(
        text=prompt,
        images=images,
        padding=True,
        return_tensors="pt",
        return_for_text_completion=True,
    ).to(model.device, dtype=model.dtype)
    
    logger.info("Generating response...")
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs,
            multimodal_generation_mode="interleaved-text-image",
            max_new_tokens=1,
            do_sample=True,
        )
    logger.info("Finished generation.")

    full_outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.exists(full_outputs_dir):
        logging.info(f"Creating directory: {full_outputs_dir}")
        os.mkdir(full_outputs_dir)
    
    # attention
    new_input_ids = output_token_ids_batch.squeeze(1).to(model.device, dtype=inputs["input_ids"].dtype).detach()
    with torch.no_grad():
        attention_output = model.forward(input_ids=new_input_ids, output_attentions=True, return_dict=True)
    attention_weights = attention_output['attentions']
    # os.makedirs(f'{full_outputs_dir}/weights', exist_ok=True)
    # torch.save(attention_weights, f'{full_outputs_dir}/weights/{data_id}.pt')
    average_attention_per_layer = []
    for layer_attention in attention_weights:
        layer_attention_numpy = layer_attention.cpu().numpy()
        average_attention = layer_attention_numpy.sum(axis=1)
        average_attention_per_layer.append(average_attention)
    average_attention_per_layer = average_attention_per_layer
    all_layers_average = np.mean(np.stack(average_attention_per_layer), axis=0)
    overall_average_attention = all_layers_average[0, :, :]
    os.makedirs(f"{outputs_dir}/{attention_dir}",exist_ok=True)
    attention_path = f'{outputs_dir}/{attention_dir}/'+id+'.png'
    plt.figure(figsize=(50, 50))
    plt.imshow(overall_average_attention, cmap='inferno', vmin=0, vmax=0.15)
    plt.tight_layout()
    plt.savefig(attention_path, bbox_inches='tight')
    plt.close()

    return attention_path

def run_text_only_generation(
    prompt: Optional[str] = None,
    image_paths: list = [],
    max_new_tokens: int = 40,
) -> str:
    
    images = [load_image(image_path) for image_path in image_paths]
    # logger.info("Images loaded.", image_1_path, image_2_path)

    inputs = processor(
        text=prompt,
        images=images,
        padding=True,
        return_tensors="pt",
        return_for_text_completion=True,
    ).to(model.device, dtype=model.dtype)

    # logger.info("Generating response...")
    with torch.inference_mode():
        output_token_ids_batch = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
    # logger.info(f"Finished generation.")

    response_token_ids = [
        output_token_ids[len(input_token_ids) :]
        for input_token_ids, output_token_ids in zip(
            inputs["input_ids"], output_token_ids_batch
        )
    ]
    response = processor.decode(response_token_ids[0], skip_special_tokens=True)

    return response


torch.set_printoptions(threshold=10_000)

model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
    attn_implementation="eager",
)

peft_model_path = "../ManyAlignment/mm_training/outputs/ft/Chameleon/dpo_Lora_radn-64-128-0.05_24-10-11-08_00_37_XP3/checkpoint-last/adapter"
model = PeftModel.from_pretrained(model, peft_model_path)

processor = ChameleonProcessor.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    token=os.environ.get("HF_TOKEN"),
)
os.makedirs("./outputs",exist_ok=True)

for dataset in ['vist']:
    processed_data = get_question_answer_dataset(dataset,"./outputs","response.json")

    for d in tqdm(processed_data):
        id = d['id']

        prompt = d['text']
        images = d['images']
        
        attention_path = run_interleaved_generation(
            data_id=id,
            prompt=prompt,
            image_paths=images,
            outputs_dir = "./outputs",
            attention_dir = "dpo_attentions"
        )