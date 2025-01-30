import numpy as np
from skimage import measure
from src.scripts.utils import read_config


# config = read_config()
# N_POINTS = config['n_points']

def img_2_points(pred, N_POINTS):
    """ Find contours and choose n points """
    # 
    print("----------------------------------------------------------------------")
    # Find contours at a constant value of 0.8
    try:
        contours = measure.find_contours(pred)[0]   # First index because it's list
        # Calculate step for cycle
        step = len(contours) // N_POINTS
        # Find coordinates of points 
        points_x = [ int(contours[i, 1]) for i in range(0, len(contours) - 1, step) ]
        points_y = [ int(contours[i, 0]) for i in range(0, len(contours) - 1, step) ]
    except:
        print("We can't obtrain contours of predicted mask")
        points_x = [ pred.shape[0] // 2 for _ in range(N_POINTS) ] # [ WINDOW_WIDTH // 2 for _ in range(N_POINTS) ]
        points_y = [ pred.shape[1] // 2 for _ in range(N_POINTS) ]  # [ WINDOW_HEIGHT // 2 for _ in range(N_POINTS) ]
        print(points_x[:5], points_y[:5])

    return np.array(points_x), np.array(points_y)


# def postprocess(predict, threshold=0.5):
#     """ Resized predicted image into the main plot shape """
#     print(f"postprocess input shape:  {predict.shape}")
#     predict = predict[0].detach().numpy()  # shape is torch.Size([1, 1, 512, 512])
#     # Reshape predict 
#     predict = np.permute_dims(predict, (1, 2, 0))
#     # Cut batch size and channels
#     # predict = predict[0]  # (h, w, n_cls)
#     if threshold != 0:
#         predict = np.where(predict >= threshold, 1, 0)
#     print(f"Postprocess output shape:  {predict.shape}")
#     return predict