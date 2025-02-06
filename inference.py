from app import get_image_from_bytes, get_bytes_from_image, get_model_predict
from app import transform_predict_to_image, transform_predict_to_dict   # add_bboxs_on_img


file = open('./src/tests/test_image.jpg', 'rb')
input_image = get_image_from_bytes(file)
predict = get_model_predict(input_image)

# print(predict)

# image = transform_predict_to_image(predict)
# Save image to results/