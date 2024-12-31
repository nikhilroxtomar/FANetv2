
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from model import Model
from utils import create_dir, seeding, calculate_metrics, init_mask, rle_encode, rle_decode
from train import load_data

def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def print_score(metrics_score):
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)
    hd = metrics_score[6]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - HD: {hd:1.4f}")

def evaluate(model, save_path, test_x, test_m, test_y, size):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    output_mask = []

    for i, (x, m, y) in tqdm(enumerate(zip(test_x, test_m, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask = mask * 255
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        """ Prev mask """
        pmask = prev_masks[i]
        pmask = " ".join(str(d) for d in pmask)
        pmask = str(pmask)
        pmask = rle_decode(pmask, size)
        pmask = np.expand_dims(pmask, axis=0)
        pmask = np.expand_dims(pmask, axis=0)
        pmask = pmask.astype(np.float32)
        pmask = torch.from_numpy(pmask)
        pmask = pmask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            _, y_pred, [h1, h2] = model(image, pmask, heatmap=True)
            y_pred = torch.sigmoid(y_pred)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Save mask """
            pred_m = y_pred[0][0].cpu().numpy()
            pred_m = pred_m > 0.5
            pred_m = np.transpose(pred_m, (1, 0))
            pred_m = np.array(pred_m, dtype=np.uint8)
            pred_m = rle_encode(pred_m)
            output_mask.append(pred_m)

            """ Evaluation metrics """
            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            """ Predicted Mask """
            y_pred = process_mask(y_pred)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", y_pred)
        cv2.imwrite(f"{save_path}/h1/{name}.jpg", h1)
        cv2.imwrite(f"{save_path}/h2/{name}.jpg", h2)

    print_score(metrics_score)
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)

    return output_mask


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "../ML_DATASET/bkai-igh-neopolyp"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    name = "bkai-igh-neopolyp"

    size = (256, 256)

    prev_masks = init_mask(test_x, size)
    for i in range(10):
        save_path = f"results/{name}/{i}"
        for item in ["mask", "joint", "h1", "h2"]:
            create_dir(f"{save_path}/{item}")

        prev_masks = evaluate(model, save_path, test_x, prev_masks, test_y, size)
