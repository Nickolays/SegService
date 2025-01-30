from PIL import Image
import torch.utils
import io, yaml, json
import pandas as pd
import numpy as np

import torch 
from typing import Optional

from src.scripts.U2Net import U2Net 
# from torch.utils.data import TensorDataset, DataLoader
from src.scripts.data_transforms import prepare_input, prepare_output
from src.scripts.utils import read_config
from src.scripts.postprocess import img_2_points


config = read_config()
img_size = (config['model_input_size'], config['model_input_size'])
device = config['device']
n_points = config['n_points']

################################# Models #####################################
# Initialize the models
model = U2Net(num_classes=2).to(device)
if device == 'cpu':
    model.load_state_dict(torch.load(config['model_path'], map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(config['model_path']))


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def get_model_predict(input_image:Image, conf:float=0.5, save:bool= alse, augment:bool=False) -> np.array:
    """
    Get the predictions of a model on an input image.
    
    Args:
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Prepare 
    input_image = prepare_input(input_image, img_size, device=device)
    # Make predictions 
    predictions = model(input_image)  # model: The trained Segmentation model.
    # Postprocess
    predictions = prepare_output(predictions, threshold=conf)
    return predictions

def transform_predict_to_dict(predict) -> json:
    """
    Transform predict from model (torch.Tensor) to json with classes and their coordinates (x, y).

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_coords (list[dict]): [{'class': class_id, class_name: "LV", coords: [[x1, y1], xn, yn]}, 
                                {'class': class_id, class_name: "LA", coords: [[...]]}]
    """

    # results = {}
    # # Convert image to points(coordinates) N_POINTS
    # for class_id, cls_name in enumerate(["LV", "LA"]):
    #     # reset
    #     outputs = {}
    #     # Get coordinates
    #     coords = img_2_points(predict[class_id], config['n_points'])
    #     # Fill current output to save in main dict
    #     outputs["class"] = class_id
    #     outputs["coords"] = coords
    #     # Save it main dict
    #     results[cls_name] = outputs.copy()

    # TODO: Change output type to list with 2 dicts
    results = []
    for class_id, cls_name in enumerate(["LV", "LA"]):
        # reset
        outputs = {}
        # Get coordinates
        x_coords, y_coords = img_2_points(predict[class_id], config['n_points'])
        # Fill current output to save in main dict
        outputs["class_id"] = class_id
        outputs["class_name"] = cls_name
        outputs["x_coords"] = x_coords.tolist()
        outputs["y_coords"] = y_coords.tolist()
        # Save it main dict
        results.append(outputs)

    return json.dumps(results)

def transform_predict_to_image(predict, treshold=0.5):
    """ Transform input predict(nunpy) to Image.Image format """
    mask = (predict[:, :, 1] > treshold)
    predict[:, :, 1][mask] = 0.5
    predict = predict.sum(axis=-1)
    predict = (predict * 255).astype(np.uint8)
    return Image.fromarray(predict)
