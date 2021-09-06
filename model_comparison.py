import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import monai
import os
from os.path import isfile, join
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import metrics

class HistoDataset(Dataset):
    """Histopathology dataset from directory containing image and mask subdirectories."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample (defaults to None).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames1 = [f for f in sorted(os.listdir(root_dir + 'image/')) if isfile(join(root_dir + 'image/', f))]
        #self.filenames2 = [f for f in (os.listdir(root_dir + 'mask/'))]

    def __len__(self):
        return len(self.filenames1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, f'image/{self.filenames1[idx]}')
        mask_path = os.path.join(self.root_dir, f'mask/{self.filenames1[idx]}')
        image = plt.imread(img_path)
        gt = plt.imread(mask_path)
        image = np.array(image)
        image = np.array(image) * 255
        # print(image.max(), image.min())
        image = image[:, :, :3]
        gt = (np.array(gt) * 255).astype(int)
        # print(gt.max(), gt.min())

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)

        # float tensor image, int tensor gt
        return image, gt, self.filenames1[idx]

# takes in 512 x 512 x 3 float tensors and returns binary tensors
def convert_mask3d(mask, img):
    """
		Converts a ground truth image (512, 512) annotated with values 0, 55, and 255
        into a binary tensor (512, 512, 3), where each layer has 1s for pixels in
        its class. Also removes pixels from the background class that are white on
        the original slide

		Parameters:
			mask - a ground truth tensor of shape (512, 512)
            img - the original slide image with shape (512, 512, 3)
	"""
    img = np.mean(img, axis=-1).astype(int)
    #new_mask0 = ((mask < 40) & (img < 220) & (img > 10)).astype(int)
    new_mask0 = ((mask < 40) & (img < 240)).astype(int)
    new_mask1 = (mask > 65).astype(int)
    new_mask2 = ((mask > 40) & (mask < 66)).astype(int)
    target_arr= np.stack((new_mask0, new_mask1, new_mask2), axis=2)
    return target_arr

# mask and gt are the two binary images -- this function calls DiceLoss monai function
def get_dice_loss(mask, gt):
    """
		Computes the dice LOSS between a mask and gt tensor using monai

		Parameters:
			mask - prediction vector with binary values in shape (512, 512)
            gt - ground truth vector with binary values in shape (512, 512)
	"""
    # if less than 800 gt pixels, return dice loss of 0
    if np.sum(gt) < 800 and np.sum(mask) == 0:
        return 0

    # convert gt and mask to one hot representation
    gt_new = np.zeros((gt.shape[0], gt.shape[1], 2))
    mask_new = np.zeros((mask.shape[0], mask.shape[1], 2))
    gt_new[gt == 0] = [1, 0]
    gt_new[gt == 1] = [0, 1]
    mask_new[mask == 0] = [1, 0]
    mask_new[mask == 1] = [0, 1]
    with torch.no_grad():
        gt_new = torch.from_numpy(gt_new)
        mask_new = torch.from_numpy(mask_new)
    gt_new = gt_new.permute(2, 0, 1).unsqueeze(0)
    mask_new = mask_new.permute(2, 0, 1).unsqueeze(0)
    criterion = monai.losses.DiceLoss(include_background=False)
    return round(criterion(mask_new, gt_new).item(), 4)

def get_binary_predictions(pred, gt):
    """
		Determines which classes (background, normal, precancerous) are present
        in both the prediction tensor and the gt tensor, and returns vectors
        containing this information to be used in computing the confusion matrix

		Parameters:
			pred - a model's prediction tensor with binary values in shape (3, 512, 512)
            gt - a binary gt vector with shape (512, 512, 3), output from convert_mask3d

        Return:
            pred_vec - vector of length 3 containing 1 if class is present in the prediction
            gt_vec - vector of length 3 containing 1 if class is present in gt
	"""
    gt_vec = []
    pred_vec = []

    # keep the threshold consistent; use one for gt and pred
    pixel_threshold = 1000 # 0.5% of slide area
    ratio_threshold = 1/8
    classes = 3

    for i in range(classes):
        gt_pos_count = np.count_nonzero(gt[:, :, i])
        pred_pos_count = np.count_nonzero(pred[i, :, :])
        print(f"Class {i+1}, y_pred_pixels: {pred_pos_count}, y_pixels: {gt_pos_count}")
        gt_vec.append(int(gt_pos_count > pixel_threshold))

        # pred must contain a large enough classification (remove ratio threshold if this is not desired)
        if gt_pos_count > pixel_threshold:
            pred_vec.append(int(pred_pos_count > gt_pos_count * ratio_threshold))
        else:
            pred_vec.append(int(pred_pos_count > pixel_threshold))
    return np.asarray(pred_vec), np.asarray(gt_vec)

def evaluate_model(model, case_names, save_path):
    """
		Evaluates a model on a set of cases and saves data analytics and roc
        curve images to the directory ./data/save_path/

		Parameters:
			model - PyTorch model to evaluate
            case_names - tuple of case_names to evaluate on
            save_path - string of the model name to save under
            batch_sz - optional parameter to evaluate the model on batches larger than 1.
                    Using a value higher than 1 may require more memory
	"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_patches = 8000 #30000
    pixels_per_patch = 2**18
    # pixel_pred_np = np.empty((pixels_per_patch * max_patches, 1), dtype='float')
    # pixel_gt_np = np.empty((pixels_per_patch * max_patches, 1), int)

    pixel_pred = torch.empty((pixels_per_patch * max_patches, 1), dtype=torch.float32, device=device)
    pixel_gt = torch.empty((pixels_per_patch * max_patches, 1), dtype=torch.int32, device=device)

    dice_data = np.empty((max_patches, 3), dtype='float')
    y_pred_np = np.empty((max_patches, 3), int)
    y_np = np.empty((max_patches, 3), int)
    curr_patch = 0
    skipped_patches = 0
    pixel_threshold = 1000

    for case_name in case_names:
        histo_dataset = HistoDataset(root_dir=f'./{case_name}/')
        dataloader = DataLoader(dataset=histo_dataset, batch_size=1, shuffle=False, num_workers=0)

        for (images, masks, patch_num) in dataloader:
            image, mask = np.squeeze(np.asarray(images)), np.squeeze(np.asarray(masks))
            gt_np = convert_mask3d(mask, image)

            normal_gt = gt_np[:, :, 1]
            precancerous_gt = gt_np[:, :, 2]
            if np.sum(precancerous_gt) < pixel_threshold and np.sum(normal_gt) < pixel_threshold:
                print(f"\n <---- Skipping background patch, {skipped_patches + 1} skipped so far ---->")
                skipped_patches += 1
                continue

            with torch.no_grad():
                images = images.to(device)
                slide_torch = images.permute(0, 3, 1, 2)
                output = model(slide_torch)
                m = nn.Sigmoid()
                output = np.squeeze(m(output).cpu().numpy())
                threshold = .5
                pred = (output > threshold).astype(int)

            print(f"\nData for {save_path}, case: {case_name}, patch {patch_num}")

            # save patch's pixel values for roc curve
            precancerous = output[2, :, :]
            ind = curr_patch * pixels_per_patch
            # if we wanted to compute roc for normal instead of precancerous:
            # normal = output[1, :, :]
            # pixel_pred_normal_np[ind:ind + pixels_per_patch] = np.reshape(normal, (pixels_per_patch, 1))
            # pixel_gt_normal_np[ind:ind + pixels_per_patch] = np.reshape(normal_gt, (pixels_per_patch, 1))
            pixel_pred[ind:ind + pixels_per_patch] = torch.from_numpy(np.reshape(precancerous, (pixels_per_patch, 1)))
            pixel_gt[ind:ind + pixels_per_patch] = torch.from_numpy(np.reshape(precancerous_gt, (pixels_per_patch, 1)))

            # collect data for monai confusion matrix
            y_pred, y = get_binary_predictions(pred, gt_np)
            y_pred_np[curr_patch, :] = y_pred
            y_np[curr_patch, :] = y

            # compute dice score for each class
            patch_dice = [1 - get_dice_loss(pred[i, :, :], gt_np[:, :, i]) for i in range(3)]
            dice_data[curr_patch, :] = patch_dice
            print("DiceScores: Class 1 (Black) =", patch_dice[0], "Class 2 (White) =", patch_dice[1], "Class 3 (Gray) =", patch_dice[2])
            print(f"y_pred: {y_pred}, y: {y}")
            curr_patch += 1
            print("CurrPatch:", curr_patch)

    # scrap the unused portion of the data arrays
    dice_data = dice_data[:curr_patch, :]
    y_pred_np = y_pred_np[:curr_patch, :]
    y_np = y_np[:curr_patch, :]
    pixel_pred = pixel_pred[:curr_patch * pixels_per_patch, :]
    pixel_gt = pixel_gt[:curr_patch * pixels_per_patch, :]
    print(dice_data.shape, y_pred_np.shape, y_np.shape, pixel_pred.shape, pixel_gt.shape)
    print("Skipped %d black patches" % skipped_patches)

    if not os.path.exists(f'./data/{save_path}'):
        os.makedirs(f'./data/{save_path}')

    print("Computing data metrics...")
    torch.cuda.empty_cache()
    rocauc = np.zeros((3, 1))
    if torch.sum(pixel_gt) != 0:
        # Change rocauc index to 1 for normal, 2 for precancerous
        with torch.no_grad():
            rocauc[2] = monai.metrics.rocauc.compute_roc_auc(pixel_pred, pixel_gt)

    # compute all metrics and concatenate them into one np.array for easy saving
    meandice = np.expand_dims(np.mean(dice_data, axis=0), axis=1)
    conf_matrix = monai.metrics.get_confusion_matrix(torch.from_numpy(y_pred_np), torch.from_numpy(y_np))
    conf_matrix = torch.sum(conf_matrix, axis=0).float()
    metric_names = ["sensitivity", "specificity", "precision", "negative predictive value", "f1_score"]
    metric_list = [monai.metrics.confusion_matrix.compute_confusion_matrix_metric(name, conf_matrix).numpy() for name in metric_names]
    metric_list = [np.expand_dims(metric, axis=1) for metric in metric_list]
    analytics = np.concatenate((meandice, conf_matrix, metric_list[0], metric_list[1], metric_list[2], metric_list[3], metric_list[4], rocauc), axis=1)

    # save analytics to csv
    I = pd.Index(['Background', 'Normal', 'Precancerous'], name="rows")
    C = pd.Index(['MeanDice', 'TP', 'FP', 'TN', 'FN', 'SEN', 'SPEC', 'PREC', 'NPV', 'F1', 'ROC-AUC'], name="cols")
    df = pd.DataFrame(analytics, index=I, columns=C)
    df.to_csv(f'./data/{save_path}/{save_path}_data.csv')

    # create roc curve, save it, then display data & roc curve
    fpr, tpr, thresholds = metrics.roc_curve(pixel_gt.cpu().numpy(), pixel_pred.cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=save_path)
    display.plot()
    plt.savefig(f'./data/{save_path}/{save_path}_roc.png')
    print(df)
    plt.show()

# same as evaluate_model, but collects data for multiple models in one call
def compare_models(models, case_names, save_paths):
    for (model, save_path) in zip(models, save_paths):
            evaluate_model(model, case_names, save_path)
