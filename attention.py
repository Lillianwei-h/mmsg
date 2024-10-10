import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('outputs2/response.json','r') as f:
    data = json.load(f)

for d in tqdm(data):
    id = d['id']
    weight_path = 'outputs2/weights/'+id+'.pt'
    attention_weights = torch.load(weight_path)

    average_attention_per_layer = []
    # 这里可以改成固定层数的
    for layer_attention in attention_weights:
        layer_attention_numpy = layer_attention.cpu().numpy()
        average_attention = layer_attention_numpy.sum(axis=1)
        average_attention_per_layer.append(average_attention)
    average_attention_per_layer = average_attention_per_layer
    all_layers_average = np.mean(np.stack(average_attention_per_layer), axis=0)

    # 这里裁切根据模型prompt和image token长度来，裁切成[prompt+response, text_prompt_end:]
    overall_average_attention = all_layers_average[0, :, :]
    attention_path = 'outputs2/attentions2/'+id+'.png'
    plt.figure(figsize=(50, 50))
    plt.imshow(overall_average_attention, cmap='inferno', vmin=0, vmax=0.1)
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(attention_path, bbox_inches='tight')
    # d['attention'] = attention_path
    plt.close()
    del attention_weights  # 或者使用 attention_weights = None
    torch.cuda.empty_cache()

# with open('outputs/response_w_attention.json','w') as f:
#     json.dump(data,f,indent=4)