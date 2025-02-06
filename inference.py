from app import get_image_from_bytes, get_bytes_from_image, get_model_predict
from app import transform_predict_to_image, transform_predict_to_dict   # add_bboxs_on_img
from PIL import Image


filename = './src/tests/test_img.jpg'
input_image = Image.open(filename).convert("RGB")
predict = get_model_predict(input_image)


image = transform_predict_to_image(predict)
image.save("results/predict.jpg")
# Save image to results/