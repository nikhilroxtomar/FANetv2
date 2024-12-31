
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from utils import mask_to_bbox, init_mask, rle_encode, rle_decode
from model import Model
from metrics import DiceLoss, DiceBCELoss

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "train", "*")))
    masks = sorted(glob(os.path.join(path, "train_gt", "*")))
    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        mask = mask * 255
        mask = mask.astype(np.uint8)

        """ Classification Label """
        num_polyps, small, medium, large = self.get_classification_label(mask)

        num_polyps = np.expand_dims(np.array(num_polyps), axis=-1)
        small = np.expand_dims(np.array(small), axis=-1)
        medium = np.expand_dims(np.array(medium), axis=-1)
        large = np.expand_dims(np.array(large), axis=-1)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255

        return [image, mask], [num_polyps, small, medium, large]

    def get_classification_label(self, label):
        """ Mask to bboxes """
        bboxes = mask_to_bbox(label)

        """ Classification: one or many """
        num_polyps = 0 if len(bboxes) == 1 else 1

        """ Classification: size -> small, medium and large """
        small, medium, large, = 0, 0, 0
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            h = (y2 - y1)
            w = (x2 - x1)
            area = (h * w) / (label.shape[0] * label.shape[1])

            if area < 0.10:
                small = 1
            elif area >= 0.10 and area < 0.30:
                medium = 1
            elif area >= 0.30:
                large = 1

        return [num_polyps, small, medium, large]

    def __len__(self):
        return self.n_samples

def train(model, loader, mask, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    return_mask = []

    for i, ([x, y], [num_polyps, small, medium, large]) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        num_polyps = num_polyps.to(device, dtype=torch.float32)
        small = small.to(device, dtype=torch.float32)
        medium = medium.to(device, dtype=torch.float32)
        large = large.to(device, dtype=torch.float32)

        # print(x.shape, y.shape)
        # print(num_polyps.shape, small.shape, medium.shape, large.shape)

        b, c, h, w  = y.shape
        m = []
        for edata in mask[i*b : i*b+b]:
            edata = " ".join(str(d) for d in edata)
            edata = str(edata)
            edata = rle_decode(edata, size)
            edata = np.expand_dims(edata, axis=0)
            m.append(edata)

        m = np.array(m, dtype=np.int32)
        m = np.transpose(m, (0, 1, 3, 2))
        m = torch.from_numpy(m)
        m = m.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        [p1, p2, p3, p4], p5 = model(x, m)
        # print(p1.shape, p2.shape, p3.shape, p4.shape, p5.shape)

        loss1 = loss_fn[0](p1, num_polyps)
        loss2 = loss_fn[0](p2, small)
        loss3 = loss_fn[0](p3, medium)
        loss4 = loss_fn[0](p4, large)
        loss5 = loss_fn[1](p5, y)
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, p5):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

            with torch.no_grad():
                yp = torch.sigmoid(yp)
                yp = yp.cpu().numpy()

                for item in yp:
                    py = np.array(item > 0.5, dtype=np.uint8)
                    py = rle_encode(py)
                    return_mask.append(py)

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision], return_mask

def evaluate(model, loader, mask, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    return_mask = []

    with torch.no_grad():
        for i, ([x, y], [num_polyps, small, medium, large]) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            num_polyps = num_polyps.to(device, dtype=torch.float32)
            small = small.to(device, dtype=torch.float32)
            medium = medium.to(device, dtype=torch.float32)
            large = large.to(device, dtype=torch.float32)

            # print(x.shape, y.shape)
            # print(num_polyps.shape, small.shape, medium.shape, large.shape)

            b, c, h, w  = y.shape
            m = []
            for edata in mask[i*b : i*b+b]:
                edata = " ".join(str(d) for d in edata)
                edata = str(edata)
                edata = rle_decode(edata, size)
                edata = np.expand_dims(edata, axis=0)
                m.append(edata)

            m = np.array(m, dtype=np.int32)
            m = np.transpose(m, (0, 1, 3, 2))
            m = torch.from_numpy(m)
            m = m.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            [p1, p2, p3, p4], p5 = model(x, m)
            # print(p1.shape, p2.shape, p3.shape, p4.shape, p5.shape)

            loss1 = loss_fn[0](p1, num_polyps)
            loss2 = loss_fn[0](p2, small)
            loss3 = loss_fn[0](p3, medium)
            loss4 = loss_fn[0](p4, large)
            loss5 = loss_fn[1](p5, y)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, p5):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

                with torch.no_grad():
                    yp = torch.sigmoid(yp)
                    yp = yp.cpu().numpy()

                    for item in yp:
                        py = np.array(item > 0.5, dtype=np.uint8)
                        py = rle_encode(py)
                        return_mask.append(py)

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision], return_mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files/checkpoint.pth"
    path = "../ML_DATASET/bkai-igh-neopolyp"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    # train_x, train_y = shuffling(train_x[:100], train_y[:100])
    # train_x = train_x[:100]
    # train_y = train_y[:100]
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    # create_dir("data")
    # for i, (x, y) in enumerate(train_dataset):
    #     x = np.transpose(x, (1, 2, 0)) * 255
    #     y = np.transpose(y, (1, 2, 0)) * 255
    #     y = np.concatenate([y, y, y], axis=-1)
    #     cv2.imwrite(f"data/{i}.png", np.concatenate([x, y], axis=1))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = Model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = [nn.CrossEntropyLoss(), DiceBCELoss()]
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    """ Init Mask """
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics, return_train_mask = train(model, train_loader, train_mask, optimizer, loss_fn, device)
        valid_loss, valid_metrics, return_valid_mask = evaluate(model, valid_loader, valid_mask, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

            train_mask = return_train_mask
            valid_mask = return_valid_mask

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
