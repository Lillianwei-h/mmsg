import os
from typing import Optional
import torch
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    set_seed,
)
from mmsg.utils import load_image
from prompts import get_ask_prompt
from dataset import get_question_dataset
from tqdm import tqdm
import json
from term_image.image import from_file
from mmsg.integrations.chameleon_utils import postprocess_token_sequence
import logging
import numpy as np
import matplotlib.pyplot as plt

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
    max_new_tokens: int = 2400,
    outputs_dir: str = "./outputs",
) -> str:
    images = [load_image(image_path) for image_path in image_paths]
    inputs = processor(
        text=prompt,
        # text = "draw a snow man and describe it",
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
    logger.info("Finished generation.")

    full_outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.exists(full_outputs_dir):
        logging.info(f"Creating directory: {full_outputs_dir}")
        os.mkdir(full_outputs_dir)
    os.makedirs(f'{full_outputs_dir}/weights', exist_ok=True)

    new_input_ids = output_token_ids_batch.squeeze(1).to(model.device, dtype=inputs["input_ids"].dtype).detach()
    with torch.no_grad():
        attention_output = model.forward(input_ids=new_input_ids, output_attentions=True, return_dict=True)
    attention_weights = attention_output['attentions']
    torch.save(attention_weights, f'{full_outputs_dir}/weights/{data_id}.pt')

    average_attention_per_layer = []
    for layer_attention in attention_weights:
        layer_attention_numpy = layer_attention.cpu().numpy()
        average_attention = layer_attention_numpy.sum(axis=1)
        average_attention_per_layer.append(average_attention)
    average_attention_per_layer = average_attention_per_layer
    all_layers_average = np.mean(np.stack(average_attention_per_layer), axis=0)
    overall_average_attention = all_layers_average[0, :, :]
    os.makedirs(f"{outputs_dir}/attentions",exist_ok=True)
    attention_path = f'{outputs_dir}/attentions/'+id+'.png'
    plt.figure(figsize=(50, 50))
    plt.imshow(overall_average_attention, cmap='inferno', vmin=0.001, vmax=0.4)
    plt.tight_layout()
    plt.savefig(attention_path, bbox_inches='tight')
    plt.close()
    
    output_token_ids_batch = output_token_ids_batch.to(dtype=inputs["input_ids"].dtype).detach().cpu().numpy()

    response_token_ids = output_token_ids_batch[0][len(inputs["input_ids"][0]) :]
    response = postprocess_token_sequence(
        response_token_ids, model, processor, full_outputs_dir, validate=True
    )
    torch.cuda.empty_cache()
    return response, attention_path

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
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
)
# model = ChameleonForConditionalGeneration.from_pretrained(
#     "leloy/Anole-7b-v0.1-hf",
#     device_map="auto",
#     token=os.environ.get("HF_TOKEN"),
#     attn_implementation="eager",
# )

processor = ChameleonProcessor.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    token=os.environ.get("HF_TOKEN"),
)
os.makedirs("./outputs2",exist_ok=True)

for dataset in ['vist']:
    processed_data = get_question_dataset(dataset)
    system_prompt = get_ask_prompt(dataset)
    if os.path.exists("./outputs2/response.json"):
        print("Find previous response!")
        with open ("./outputs2/response.json",'r') as f:
            processed_data = json.load(f)
    for d in tqdm(processed_data):
        if "answer" in d:
            continue
        id = d['id']
        question = d['question_text']
        images = d['images']
        prompt = system_prompt.format(question=question)
        
        response, attention_path = run_interleaved_generation(
            data_id=id,
            prompt=prompt,
            image_paths=images,
            outputs_dir = "./outputs2"
        )
        d["answer"] = response
        d["attention"] = attention_path
        d.pop('question_text')
        d.pop('images')
        d['model'] = "anole"
        with open(f'./outputs2/response.json','w') as f:
            json.dump(processed_data, f, indent=4)

    with open(f'./outputs2/response.json','w') as f:
        json.dump(processed_data, f, indent=4)