import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np

CHECKPOINT_PATH = "sam_vit_l_0b3195.pth"
MODEL_TYPE = "vit_l"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)
img_path = "dog_image.jpg"
img = Image.open(img_path)
img_array = np.array(img)

predictor.set_image(img_array)
masks, _, _ = predictor.predict()


