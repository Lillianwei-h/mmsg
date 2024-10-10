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

def tokenization(
    prompt: Optional[str] = None,
    image_paths: list = [],
) -> str:
    images = [load_image(image_path) for image_path in image_paths]

    # self.image_token = <image> 8711
    # self.image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode 8197
    # self.image_end_token = "<eoss>" 8196
    # one_img_tokens = self.image_start_token + (self.image_token * self.image_seq_length) + self.image_end_token
    inputs = processor(
        text=prompt,
        images=images,
        padding=True,
        return_tensors="pt",
        return_for_text_completion=True,
    )
    token_ids = inputs["input_ids"]
    return token_ids


torch.set_printoptions(threshold=10_000)

processor = ChameleonProcessor.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    token=os.environ.get("HF_TOKEN"),
)
os.makedirs("./output_tokens",exist_ok=True)

for dataset in ['vist']:
    processed_data = get_question_anser_dataset(dataset)
    for d in tqdm(processed_data[:5]):
        id = d['id']
        question = d['question_text']
        question_answer = d['question_anwer_text']
        question_images = d['question_images']
        question_anwer_images = d['question_anwer_images']
        
        question_token = tokenization(
            prompt=question,
            image_paths=question_images
        )
        
        question_answer_token = tokenization(
            prompt=question_answer,
            image_paths=question_anwer_images
        )

        d["question_token"] = question_token
        d["question_answer_token"] = question_answer_token
        d.pop('question_text')
        d.pop('question_anwer_text')
        d.pop('question_images')
        d.pop('question_anwer_images')

        with open(f'./output_tokens/tokenization.json','w') as f:
            json.dump(processed_data, f, indent=4)

    with open(f'./output_tokens/tokenization.json','w') as f:
        json.dump(processed_data, f, indent=4)