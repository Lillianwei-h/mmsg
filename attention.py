import torch

# inputs = torch.load('input.pt')
output = torch.load('attention.pt')

import numpy as np
import matplotlib.pyplot as plt

attention_weights = output['attentions']

average_attention_per_layer = []

# 这里可以改成固定层数的
for layer_attention in attention_weights:
    layer_attention_numpy = layer_attention.cpu().numpy()
    average_attention = layer_attention_numpy.sum(axis=1)
    average_attention_per_layer.append(average_attention)
average_attention_per_layer = average_attention_per_layer
all_layers_average = np.mean(np.stack(average_attention_per_layer), axis=0)

# 这里裁切根据模型prompt和image token长度来，裁切成[prompt+response, text_prompt_end:]
overall_average_attention = all_layers_average[0, 1037:, :]

plt.figure(figsize=(50, 50))
plt.imshow(overall_average_attention, cmap='inferno', vmin=0.001, vmax=0.4)
# plt.colorbar()
plt.tight_layout()
plt.savefig('image2.png', bbox_inches='tight')
# plt.show()