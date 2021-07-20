# python cityscapes_semanticsegmentation.py 
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import sys

def semantic_segmentation(filename):

  input_image = Image.open(filename).convert('RGB')
  preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0)

  if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
  else:
    input_batch = input_batch.to('cpu')
    model.to('cpu')

  with torch.no_grad():
    outputs = model(input_batch)['out'][0]
    print(outputs.shape)
  
  output_predictions = outputs.argmax(0)
  print(output_predictions.shape)

  palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
  colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
  colors = (colors % 255).numpy().astype("uint8")
  

  # for i in range(21):
  #   colors[i][0] = 0
  #   colors[i][1] = 0
  #   colors[i][2] = 0
  r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
  r.putpalette(colors)

  if r.mode != "RGB":
    r = r.convert("RGB")
  r.save("test_output.png")

if __name__ == "__main__":
  model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
  model.load_state_dict(torch.load("best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar", map_location='cpu')['model_state'], strict=False)
  model.eval()

  # labels = [[0, ], [1, ], [2, ], [3, ], [4, ], [5, ], [6, ], [7, ], [8, ], [9, ], [10, ], [11, ], [12, ], [13, ], [14, ], [15, ], [16, ], [17, ], [18, ], [19, ], [20, ]]

  args = sys.argv
  semantic_segmentation(args[1])
