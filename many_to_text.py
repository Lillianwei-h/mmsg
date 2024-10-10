import os
from typing import Optional
import torch
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    set_seed,
)
from mmsg.utils import load_image
from prompts import get_prompt
from dataset import get_dataset
from tqdm import tqdm
import json



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
    # logger.info(f"Response: {response}")
    # print(response)
    return response


torch.set_printoptions(threshold=10_000)

model = ChameleonForConditionalGeneration.from_pretrained(
    "/home/siweih/Project/models/Anole-7b-v0.1-hf",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
)
processor = ChameleonProcessor.from_pretrained(
    "/home/siweih/Project/models/Anole-7b-v0.1-hf",
    token=os.environ.get("HF_TOKEN"),
)

for dataset in ['mathvista','remi','vist','wikihow']:
    processed_data = get_dataset(dataset)
    system_prompt = get_prompt(dataset)
    for d in tqdm(processed_data):
        question = d['question']
        answer = d['answer']
        images = d['images']
        if dataset in ['mathvista']:
                gt_answer = d['gt_answer']
                prompt = system_prompt.format(question=question,answer=answer,gt_answer=gt_answer)
        else:
            prompt = system_prompt.format(question=question,answer=answer)

        response=run_text_only_generation(
            prompt=prompt,
            image_paths=images,
            max_new_tokens=500,
        )
        d["gpt_feedback"] = response
        with open(f'{dataset}_eval_result_temp.json','w') as f:
            json.dump(processed_data, f, indent=4)

    with open(f'{dataset}_eval_result.json','w') as f:
        json.dump(processed_data, f, indent=4)