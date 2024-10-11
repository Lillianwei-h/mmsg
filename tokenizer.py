import os
from typing import Optional
import torch
from transformers import ChameleonProcessor, ChameleonModel, ChameleonForConditionalGeneration
from utils import load_image
from dataset import get_question_answer_dpo_dataset
from tqdm import tqdm
import json
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

def tokenization(
    prompt: Optional[str] = None,
    image_paths: list = [],
):

    if len(image_paths) == 0:
        inputs = processor(
            text=prompt,
            padding=True,
            return_tensors="pt",
            return_for_text_completion=True,
        ).to(model.device, dtype=model.dtype)
        return inputs["input_ids"]

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
    ).to(model.device, dtype=model.dtype)
    combine_ids = model.combine_ids(inputs["pixel_values"],inputs["input_ids"])
    return combine_ids


torch.set_printoptions(threshold=10_000)
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
)
processor = ChameleonProcessor.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    token=os.environ.get("HF_TOKEN"),
)
os.makedirs("./output_tokens",exist_ok=True)

for dataset in ['vist']:
    processed_data = get_question_answer_dpo_dataset(dataset,"gpt_flux_results","anole_select_gpt")
    tokens = []
    for d in tqdm(processed_data):
        id = d['id']

        question = d['question_text']
        answer1 = d['answer1_text']
        answer2 = d['answer2_text']

        question_images = d['question_images']
        answer1_images = d['answer1_images']
        answer2_images = d['answer2_images']
        
        question_token = tokenization(
            prompt=question,
            image_paths=question_images
        )

        answer1_token = tokenization(
            prompt=answer1,
            image_paths=answer1_images
        )

        answer2_token = tokenization(
            prompt=answer2,
            image_paths=answer2_images
        )

        d["question_token"] = question_token.tolist()[0]
        d["answer1_token"] = answer1_token.tolist()[0]
        d["answer2_token"] = answer2_token.tolist()[0]

        d.pop('question_text')
        d.pop('answer1_text')
        d.pop('answer2_text')
        d.pop('question_images')
        d.pop('answer1_images')
        d.pop('answer2_images')

        tokens.append(
            {
                "question": d["question_token"],
                "selected": d["answer1_token"],
                "rejected": d["answer2_token"]
            }
        )
        with open('./output_tokens/dpo_token_gptf_anole.json','w') as f:
            json.dump(tokens, f)
    with open('./output_tokens/dpo_token_gptf_anole.json','w') as f:
        json.dump(tokens, f)