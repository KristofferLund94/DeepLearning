import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from PIL import ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001 
num_epochs = 30
max_gradcam_images = 100
gradcam = True


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.hook_gradients()
        self.hook_activations()

    def hook_gradients(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_backward_hook(backward_hook)

    def hook_activations(self):
        def forward_hook(module, input, output):
            self.activations = output
        self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        # Enable gradient computation
        with torch.set_grad_enabled(True):
            # Forward pass
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            output[:, target_class].backward()

            # Compute Grad-CAM
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling
            cam = (weights * self.activations).sum(dim=1).squeeze()
            cam = torch.relu(cam)  # ReLU to ensure non-negativity
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize

        return cam

def generate_and_save_gradcam(model, input_tensor, target_layers, original_image, output_dir, target_classes=None, label='', predicted_label=''):

    os.makedirs(output_dir, exist_ok=True)

    if target_classes is None:
        target_classes = range(model.fc[-1].out_features)  # Assuming last FC layer defines num_classes

    for layer in target_layers:
        gradcam = GradCAM(model, layer)
        for target_class in target_classes:
            cam = gradcam.generate_cam(input_tensor, target_class)

            # Prepare the original image
            original_image_normalized = original_image.astype(np.float32) / 255.0 
            overlayed_image = apply_colormap_on_image((original_image * 255).astype(np.uint8), cam)

            # Save Grad-CAM visualizations
            filename = f"gradcam_layer-{layer}_class-{target_class}.png"
            filepath = os.path.join(output_dir, filename)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image\nground truth: " + str(label))
            plt.imshow(original_image_normalized)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title(f"Grad-CAM (Class {target_class})\npredicted class: " + str(predicted_label))
            plt.imshow(overlayed_image)
            plt.axis('off')

            plt.savefig(filepath)
            plt.close()
            print(f"Saved Grad-CAM for class {target_class} at {filepath}")



def visualize_with_gradcam(model, dataset, transform, output_dir, target_layers, device):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    processed_images = 0

    with torch.no_grad():
        for idx, (image, label) in enumerate(dataset):
            if processed_images >= max_gradcam_images:
                break  # Stop once the maximum number of images is reached
            # Ensure we work with the raw, original image
            if isinstance(image, torch.Tensor):
                original_image = to_pil_image(image)  # Convert raw tensor to PIL Image
            else:
                original_image = image  # Use raw image directly if already a PIL Image

            # Preprocess for the model (normalize)
            image_tensor = transform(original_image).unsqueeze(0).to(device)

            # Get model predictions
            outputs = model(image_tensor)
            predicted_label = outputs.argmax(dim=1).item()  # Predicted class

            # Convert the original image to numpy for overlay visualization
            original_image_np = np.asarray(original_image)

            # Create a folder with both ground truth and predicted labels
            output_subdir = os.path.join(
                output_dir,
                f"image-{idx}_pred-{predicted_label}_gt-{label}"
            )

            # Generate Grad-CAM visualizations for all classes
            generate_and_save_gradcam(
                model, image_tensor, target_layers, original_image_np, output_subdir, target_classes=None, label=label, predicted_label=predicted_label
            )
            
            processed_images += 1

def apply_colormap_on_image(image, activation_map, colormap=cv2.COLORMAP_JET):
    """
    Apply a colormap to the activation map and overlay it on the image.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlayed_image

class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, preprocessing=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.preprocessing = preprocessing 
        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    def preprocess_image(self, img):
        if self.preprocessing is None:
            return img

        if 'ben_graham' in self.preprocessing:
            img = self.ben_graham_preprocessing(img)
        if 'circle_crop' in self.preprocessing:
            img = self.circle_crop(img)
        if 'clahe' in self.preprocessing:
            img = self.apply_clahe(img)
        if 'gaussian_blur' in self.preprocessing:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.preprocessing['gaussian_blur']))
        if 'sharpen' in self.preprocessing:
            img = img.filter(ImageFilter.SHARPEN)
        return img

    def ben_graham_preprocessing(self, img):
        size = min(img.size)
        img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        img.putalpha(mask)
        img = img.crop(mask.getbbox())
        return img.convert("RGB")

    def circle_crop(self, img):
        size = min(img.size)
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        result = Image.new('RGB', (size, size), (0, 0, 0))
        result.paste(img, (0, 0), mask)
        return result

    def apply_clahe(self, img):
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')

        img = self.preprocess_image(img) # Preprocessing before transform

        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        img1 = self.preprocess_image(img1) # Preprocessing before transform
        img2 = self.preprocess_image(img2) # Preprocessing before transform

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]
        


class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img

transform_train_experiment = transforms.Compose([
    transforms.Resize((256, 256)),                          
    transforms.RandomCrop((210, 210)),                      
    SLORandomPad((224, 224)),                               # Add padding to restore final size
    FundRandomRotate(prob=0.5, degree=15),                  # Smaller rotation for medical images
    transforms.RandomHorizontalFlip(p=0.5),                # Consider removing if orientation matters
    transforms.ColorJitter(brightness=(0.7, 1.3)),          # Subtle brightness variation
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_train_cutoff = transforms.Compose([
    transforms.Resize((256, 256)),                          
    transforms.RandomCrop((210, 210)),                      # Retain more information
    SLORandomPad((224, 224)),                               # Add padding to restore final size
    FundRandomRotate(prob=0.5, degree=15),                  # Smaller rotation for medical images
    transforms.RandomHorizontalFlip(p=0.5),                # Consider removing if orientation matters
    transforms.ColorJitter(brightness=(0.8, 1.2)),          # Subtle brightness variation
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    CutOut(mask_size=16, p=0.5)                             # Add CutOut 
])

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train_aggressive = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 1.0)),
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    CutOut(mask_size=100, p=0.5)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_visualize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_normalize =transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def plot_training_metrics(training_kappas, validation_kappas, training_accuracies, validation_accuracies, training_precisions, validation_precisions, training_recalls, validation_recalls, save_path):
    epochs = range(1, len(training_kappas) + 1)

    # Kappa Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_kappas, label='Training Kappa', marker='o')
    plt.plot(epochs, validation_kappas, label='Validation Kappa', marker='x')
    plt.title('Kappa Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Kappa')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_kappa.png'))
    plt.close()  # Close the plot instead of displaying it

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', marker='x')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_accuracy.png'))
    plt.close()  # Close the plot instead of displaying it

    # Precision Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_precisions, label='Training Precision', marker='o')
    plt.plot(epochs, validation_precisions, label='Validation Precision', marker='x')
    plt.title('Precision Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_precision.png'))
    plt.close()  # Close the plot instead of displaying it

    # Recall Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_recalls, label='Recall Accuracy', marker='o')
    plt.plot(epochs, validation_recalls, label='Recall Accuracy', marker='x')
    plt.title('Recall Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_recall.png'))
    plt.close()  # Close the plot instead of displaying it

def plot_combined_metrics(training_kappas, validation_kappas, 
                          training_accuracies, validation_accuracies, 
                          training_precisions, validation_precisions, 
                          training_recalls, validation_recalls, save_path):
    epochs = range(1, len(training_kappas) + 1)

    plt.figure(figsize=(12, 8))

    # Kappa
    plt.plot(epochs, training_kappas, label='Training Kappa', linestyle='--', color='blue')
    plt.plot(epochs, validation_kappas, label='Validation Kappa', linestyle='-', linewidth=2, color='blue')

    # Accuracy
    plt.plot(epochs, training_accuracies, label='Training Accuracy', linestyle='--', color='green')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', linestyle='-', linewidth=2, color='green')

    # Precision
    plt.plot(epochs, training_precisions, label='Training Precision', linestyle='--', color='orange')
    plt.plot(epochs, validation_precisions, label='Validation Precision', linestyle='-', linewidth=2, color='orange')

    # Recall
    plt.plot(epochs, training_recalls, label='Training Recall', linestyle='--', color='red')
    plt.plot(epochs, validation_recalls, label='Validation Recall', linestyle='-', linewidth=2, color='red')

    # Formatting the plot
    plt.title('Training and Validation Metrics Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent display during training


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth', print_string=''):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    plot_path='./outputs/plots/training_plot' + print_string + '.png'
    training_kappas, validation_kappas = [], []
    training_accuracies, validation_accuracies = [], []
    training_precisions, validation_precisions = [], []
    training_recalls, validation_recalls = [], []

    for epoch in range(1, num_epochs + 1):
        print(print_string)
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        train_metrics = compute_metrics(all_preds, all_labels, per_class=False)
        kappa, accuracy, precision, recall = train_metrics[:4]
        training_kappas.append(kappa)
        training_accuracies.append(accuracy)
        training_precisions.append(precision)
        training_recalls.append(recall)

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        validation_kappas.append(val_kappa)
        validation_accuracies.append(val_accuracy)
        validation_precisions.append(val_precision)
        validation_recalls.append(val_recall)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)
        
        print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')
        
        plot_combined_metrics(training_kappas, validation_kappas, training_accuracies, validation_accuracies, training_precisions, validation_precisions, training_recalls, validation_recalls, plot_path)

    return model

def get_meta_model(base_model_preds, true_labels):

    # Define a logistic regression as the meta model
    meta_model = LogisticRegression()

    # Train meta model on the stacked predictions
    meta_model.fit(base_model_preds, true_labels)
    
    return meta_model


def optimize_weights(base_model_preds, true_labels):

    def weighted_predictions(weights):
        weighted_preds = np.dot(base_model_preds, weights)
        return np.round(weighted_preds)

    def loss_function(weights):
        preds = weighted_predictions(weights)
        kappa = cohen_kappa_score(true_labels, preds)
        return -kappa  # Minimize the negative kappa score

    # Initial equal weights
    initial_weights = [1 / base_model_preds.shape[1]] * base_model_preds.shape[1]

    # Define bounds and constraints
    bounds = [(0, 1) for _ in range(base_model_preds.shape[1])]  # Weights between 0 and 1
    constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}  # Sum of weights must be 1

    # Optimize weights
    from scipy.optimize import minimize
    result = minimize(loss_function, initial_weights, bounds=bounds, constraints=constraints)

    return result.x.tolist()


def bagging_ensemble(predictions):
    n_models = len(predictions)
    n_samples = len(predictions[0])

    # Simulate bootstrapping by resampling predictions
    bootstrapped_predictions = np.zeros((n_samples, n_models), dtype=int)
    for i in range(n_models):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrapped_predictions[:, i] = predictions[i][indices]

    # Aggregate predictions for each sample using majority voting
    aggregated_predictions = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        aggregated_predictions[i] = np.bincount(bootstrapped_predictions[i, :]).argmax()

    return aggregated_predictions


def max_voting(predictions):
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)


def weighted_average(predictions, true_labels):

    base_model_preds = np.column_stack(predictions)
    weights = optimize_weights(base_model_preds, true_labels)
    return np.average(base_model_preds, axis=1, weights=weights).round().astype(int)


def stacking_ensemble(predictions, true_labels):

    base_model_preds = np.column_stack(predictions)
    meta_model = get_meta_model(base_model_preds, true_labels)
    return meta_model.predict(base_model_preds)


def boosting_ensemble(predictions, true_labels):

    base_model_preds = np.column_stack(predictions)
    gbt = GradientBoostingClassifier()
    gbt.fit(base_model_preds, true_labels)
    return gbt.predict(base_model_preds)


def compare_ensembles(val_pred_paths):

    # Load true labels
    y_true = pd.read_csv("./DeepDRiD/valtargets.csv", dtype=int).iloc[:, 0].values.flatten()

    # Load predictions from CSV files and cast to integers, using the second column (targets)
    predictions_list = [
        pd.read_csv(path).iloc[:, 1].astype(int).values for path in val_pred_paths
    ]

    # Define methods
    methods = {
        "max_voting": lambda preds: max_voting(preds),
        "bagging": lambda preds: bagging_ensemble(preds),
        "weighted_average": lambda preds: weighted_average(preds, y_true),
        "stacking": lambda preds: stacking_ensemble(preds, y_true),
        "boosting": lambda preds: boosting_ensemble(preds, y_true)
    }

    results = {}
    for method_name, method_func in methods.items():
        final_predictions = method_func(predictions_list)
        print(final_predictions)
        results[method_name] = cohen_kappa_score(y_true, final_predictions)

    return results


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if 'val_pred' in prediction_path:
                images, labels = data
            else:
                if test_only:
                    images = data
                else:
                    images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall




class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        q = self.query(x)
        k = self.key(x).transpose(-2, -1)
        v = self.value(x)

        # Compute attention scores
        scores = torch.matmul(q, k) / (x.size(-1) ** 0.5)
        attention = self.softmax(scores)

        # Apply attention
        attended = torch.matmul(attention, v)
        return attended


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        avg_out = self.avg_pool(x).view(x.size(0), -1)  # [batch_size, channels]
        fc_out = self.fc2(torch.relu(self.fc1(avg_out)))
        attention = self.sigmoid(fc_out).view(x.size(0), x.size(1), 1, 1)  # [batch_size, channels, 1, 1]
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention map
        avg_out = x.mean(dim=1, keepdim=True)  # Channel average
        max_out, _ = x.max(dim=1, keepdim=True)  # Channel max
        attention = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 2, height, width]
        attention = self.sigmoid(self.conv1(attention))  # [batch_size, 1, height, width]
        return x * attention



class MyModel(nn.Module):
    def __init__(self, backbone_name="vgg16", state_dict_path='./outputs/task2/single_vgg_unfrozen_transoriginal_kaggleDRresized/model_single_vgg16_freezeFalse_transformoriginal_0.pth', attention='none', freeze_backbone=False, num_classes=5, dropout_rate=0.5):
        super().__init__()

        # Select the backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification layer

        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "vgg16":
            self.backbone = models.vgg16(pretrained=True)
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = 512  # 512 # Output size of VGG-16's features block
            self.backbone.classifier = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "densenet121":
            self.backbone = models.densenet121(pretrained=True)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Remove classification layer
            
        else:
            raise ValueError(f"Backbone '{backbone_name}' is not supported.")
        
        if state_dict_path != 'none':
            state_dict = torch.load(state_dict_path, map_location='cuda')
            info = self.backbone.load_state_dict(state_dict, strict=False)
            print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
            print('unexpected keys:', info[1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.attention = attention

        # Define the attention mechanism
        if attention == 'self':
            self.attention_layer1 = SelfAttention(256)
            self.attention_layer2 = SelfAttention(128)
        elif attention == 'channel':
            self.attention_layer1 = ChannelAttention(256)
            self.attention_layer2 = ChannelAttention(128)
        elif attention == 'spatial':
            self.attention_layer1 = SpatialAttention()
            self.attention_layer2 = SpatialAttention()

        # Define the classifier with attention after each layer
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),  # FC2
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)  # Final classification layer
        )

    def forward(self, x):
        # Pass through the backbone
        x = self.backbone(x)  # Output shape: [batch_size, 512]

        # Pass through FC1
        x = self.fc[0](x)  # Linear(512, 256)
        x = self.fc[1](x)  # ReLU
        x = self.fc[2](x)  # Dropout

        # Apply attention after FC1
        if self.attention in ['self', 'channel', 'spatial']:
            if self.attention == 'self':
                x = x.unsqueeze(1)  # Add sequence dimension for self-attention
                x = self.attention_layer1(x)
                x = x.squeeze(1)  # Remove sequence dimension
            elif self.attention == 'channel':
                x = self.attention_layer1(x.unsqueeze(-1).unsqueeze(-1))
                x = x.squeeze(-1).squeeze(-1)
            elif self.attention == 'spatial':
                x = self.attention_layer1(x.unsqueeze(-1).unsqueeze(-1))
                x = x.squeeze(-1).squeeze(-1)

        # Pass through FC2
        x = self.fc[3](x)  # Linear(256, 128)
        x = self.fc[4](x)  # ReLU
        x = self.fc[5](x)  # Dropout

        # Apply attention after FC2
        if self.attention in ['self', 'channel', 'spatial']:
            if self.attention == 'self':
                x = x.unsqueeze(1)  # Add sequence dimension for self-attention
                x = self.attention_layer2(x)
                x = x.squeeze(1)  # Remove sequence dimension
            elif self.attention == 'channel':
                x = self.attention_layer2(x.unsqueeze(-1).unsqueeze(-1))
                x = x.squeeze(-1).squeeze(-1)
            elif self.attention == 'spatial':
                x = self.attention_layer2(x.unsqueeze(-1).unsqueeze(-1))
                x = x.squeeze(-1).squeeze(-1)

        # Final classification layer
        x = self.fc[6](x)  # Linear(128, num_classes)
        return x


class MyDualModel(nn.Module):
    def __init__(self, backbone_name="vgg16", state_dict_path='./outputs/task2/single_vgg_unfrozen_transoriginal_kaggleDRresized/model_single_vgg16_freezeFalse_transformoriginal_0.pth', attention='none', freeze_backbone=False, num_classes=5, dropout_rate=0.5):
        super().__init__()

        # Select the backbone
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove classification layer

        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "vgg16":
            backbone = models.vgg16(pretrained=True)
            backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = 512  # 512 # Output size of VGG-16's features block
            backbone.classifier = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "densenet121":
            backbone = models.densenet121(pretrained=True)
            feature_dim = self.backbone.classifier.in_features
            backbone.classifier = nn.Identity()  # Remove classification layer
            
        else:
            raise ValueError(f"Backbone '{backbone_name}' is not supported.")
        
        if state_dict_path != 'none':
            state_dict = torch.load(state_dict_path, map_location='cuda')
            info = backbone.load_state_dict(state_dict, strict=False)
            print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
            print('unexpected keys:', info[1])


        self.attention = attention

        if attention == 'self':
            self.attention_layer = SelfAttention(feature_dim) # feature_dim or 512
        elif attention == 'channel':
            self.attention_layer = ChannelAttention(feature_dim) # feature_dim, or 512
        elif attention == 'spatial':
            self.attention_layer = SpatialAttention()


        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        if freeze_backbone:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False


        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)


        # Pass through the backbone
        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        # Apply attention if specified
        if self.attention == 'self':
            x1 = x1.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, 512]
            x1 = self.attention_layer(x1)  # SelfAttention expects [batch_size, seq_len, feature_dim]
            x1 = x1.squeeze(1)  # Remove the sequence dimension: [batch_size, 512]
            x2 = x2.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, 512]
            x2 = self.attention_layer(x2)  # SelfAttention expects [batch_size, seq_len, feature_dim]
            x2 = x2.squeeze(1)  # Remove the sequence dimension: [batch_size, 512]

        elif self.attention == 'channel':
            x1 = self.attention_layer(x1.unsqueeze(-1).unsqueeze(-1))  # Reshape to [batch_size, 512, 1, 1]
            x1 = x1.squeeze(-1).squeeze(-1)  # Back to [batch_size, 512]
            x2 = self.attention_layer(x2.unsqueeze(-1).unsqueeze(-1))  # Reshape to [batch_size, 512, 1, 1]
            x2 = x2.squeeze(-1).squeeze(-1)  # Back to [batch_size, 512]

        elif self.attention == 'spatial':
            # Reshape back to pseudo-spatial dimensions for SpatialAttention
            x1 = x1.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, 512, 1, 1]
            x1 = self.attention_layer(x1)
            x1 = x1.squeeze(-1).squeeze(-1)  # Back to [batch_size, 512]
            x2 = x2.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, 512, 1, 1]
            x2 = self.attention_layer(x2)
            x2 = x2.squeeze(-1).squeeze(-1)  # Back to [batch_size, 512]

        # Pass through the fully connected layers
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        
        return x

def compare_training(dataset_name = 'DeepDRiD/', state_dict_path='none', attentions=['none'], subfolder_name='default_output', backbone_names=["vgg16"], modes=['single'], backbone_freezes=[False], train_transforms={"transformoriginal": transform_train_original}, loops=6, preprocessing_options=None):
    mainfolder_path = './'
    outfolder_path = 'outputs/task4-1/'
    subfolder_path = mainfolder_path + outfolder_path + subfolder_name
    dataset_path = mainfolder_path + dataset_name
    log_file_path = subfolder_path + '/metrics_log.txt'
    os.makedirs(subfolder_path, exist_ok=True)
    

    val_metrics_dictionary = {}
    kappa_averages = {}


    # Creating hella loops for running many combinations while I sleep, makes it flexible for testing exactly what I want without having to manually start a new model training
    for mode in modes:
        for backbone_name in backbone_names:
            for freeze_backbone in backbone_freezes:
                for train_transform in train_transforms:
                    for attention in attentions:
                        kappa_scores = []
                        for loop in range(loops):
                            preproccess_name = '_none'
                            if preprocessing_options != None:
                                preproccess_name = ''
                                for option in preprocessing_options:
                                    preproccess_name += '_' + str(option)
                                    
                            model_name = mode + '_' + backbone_name + '_freeze' + str(freeze_backbone) + '_' + train_transform + '_' + attention + preproccess_name
                            loop_name = model_name + '_' + str(loop)
                            model_path = subfolder_path + '/model_' + loop_name + '.pth'
                            prediction_path = subfolder_path + '/test_predictions_' + loop_name + '.csv'
                            prediction_path_val = subfolder_path + '/val_predictions_' + loop_name + '.csv'

                            # Choose between 'single image' and 'dual images' pipeline
                            # This will affect the model definition, dataset pipeline, training and evaluation

                            # mode = 'single'  # forward single image to the model each time
                            # mode = 'dual'  # forward two images of the same eye to the model and fuse the features

                            assert mode in ('single', 'dual')

                            # Define the model
                            if mode == 'single':
                                model = MyModel(backbone_name=backbone_name, state_dict_path=state_dict_path, freeze_backbone=freeze_backbone, attention=attention)
                            else:
                                model = MyDualModel()

                            print(model, '\n')
                            print('Pipeline Mode:', mode)

                            # Create datasets
                            train_dataset = RetinopathyDataset(dataset_path + 'train.csv', dataset_path + 'train/', train_transforms[train_transform], mode=mode, preprocessing=preprocessing_options)
                            val_dataset = RetinopathyDataset(dataset_path + 'val.csv', dataset_path + 'val/', transform_test, mode=mode, preprocessing=preprocessing_options)
                            test_dataset = RetinopathyDataset(dataset_path + 'test.csv', dataset_path + 'test/', transform_test, mode=mode, test=True, preprocessing=preprocessing_options)
                            visualize_dataset = RetinopathyDataset(dataset_path + 'val.csv', dataset_path + 'val/', transform_visualize, mode=mode, preprocessing=preprocessing_options)

                            # Create dataloaders
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                            # Define the weighted CrossEntropyLoss
                            criterion = nn.CrossEntropyLoss()

                            # Use GPU device is possible
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            print('Device:', device)
                            #print(torch.cuda.get_device_name(0))

                            # Move class weights to the device
                            model = model.to(device)

                            # Optimizer and Learning rate scheduler
                            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
                            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

                            # Train and evaluate the model with the training and validation set
                            model = train_model(
                                model, train_loader, val_loader, device, criterion, optimizer,
                                lr_scheduler=lr_scheduler, num_epochs=num_epochs,
                                checkpoint_path = model_path,
                                print_string=loop_name
                            )
                            

                            state_dict = torch.load(model_path, map_location='cuda', weights_only=False)

                            model.load_state_dict(state_dict, strict=True)

                            # Make predictions on testing set and save the prediction results
                            evaluate_model(model, test_loader, device, test_only=True, prediction_path=prediction_path)

                            
                            val_metrics = evaluate_model(model, val_loader, device)
                            evaluate_model(model, val_loader, device, test_only=True, prediction_path=prediction_path_val)
                            val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
                            val_metrics_dictionary[loop_name] = {
                                'Kappa': round(val_kappa, 4),
                                'Accuracy': round(val_accuracy, 4),
                                'Precision': round(val_precision, 4),
                                'Recall': round(val_recall, 4)
                                }

                            kappa_scores.append(val_kappa)

                            with open(log_file_path, 'a') as f:
                                print(val_metrics_dictionary[loop_name])
                                print(val_metrics_dictionary[loop_name], file=f)
                            
                            # Grad-CAM Visualization
                            if gradcam:
                                print("Generating Grad-CAM visualizations...")
                                target_layers = [model.backbone.features[-1]] 
                                visualize_with_gradcam(
                                    model=model,
                                    dataset=visualize_dataset,
                                    transform=transform_test,
                                    output_dir=f"{subfolder_path}/gradcam_{loop_name}/",
                                    target_layers=target_layers,
                                    device=device
                                )
                
                        kappa_averages[model_name] = np.mean(kappa_scores)


                        with open(log_file_path, 'a') as f:
                            for kappa in kappa_averages:
                                print(f'\nAverage Kappa Scores:')
                                print(f'\nAverage Kappa Scores:', file=f)
                                print(kappa)
                                print(kappa, file=f)

                        with open(log_file_path, 'a') as f:
                            print(f'All Metrics:\n{val_metrics_dictionary}\n')
                            print(f'All Metrics:\n{val_metrics_dictionary}\n', file=f)
                            
                            print(f'Average Kappa Scores:\n{kappa_averages}\n')
                            print(f'Average Kappa Scores:\n{kappa_averages}\n', file=f)
        

def grid_search_preprocessing(
    dataset_name='DeepDRiD/',
    state_dict_path='./pretrained_DR_resize/pretrained/densenet121.pth',
    backbone='densenet121',
    mode='single',
    freeze_backbone=False,
    attention='none',
    train_transform_name="transformoriginal",
    train_transforms={"transformoriginal": transform_train_original},
    preprocessing_options_list=[
        {'ben_graham': True},
        {'circle_crop': True},
        {'clahe': True},
        {'gaussian_blur': 2},
        {'sharpen': True},
        {'ben_graham': True, 'circle_crop': True},
        {'clahe': True, 'gaussian_blur': 2, 'sharpen': True},
        {'ben_graham': True, 'circle_crop': True, 'clahe': True, 'gaussian_blur': 2, 'sharpen': True},
        None,  # No preprocessing
    ],
    loops=6,
    output_folder='preprocessing_grid_search1'
):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over preprocessing options
    for preprocessing_options in preprocessing_options_list:
        preprocessing_name = (
            "none" if preprocessing_options is None 
            else "_".join(f"{key}{val}" for key, val in preprocessing_options.items())
        )
        subfolder_name = f"{train_transform_name}_{preprocessing_name}"
        print(f"Running configuration: {subfolder_name}")

        # Call the compare_training function with the current preprocessing configuration
        compare_training(
            dataset_name=dataset_name,
            state_dict_path=state_dict_path,
            attentions=[attention],
            subfolder_name=subfolder_name,
            backbone_names=[backbone],
            modes=[mode],
            backbone_freezes=[freeze_backbone],
            train_transforms={train_transform_name: train_transforms[train_transform_name]},
            loops=loops,
            preprocessing_options=preprocessing_options
        )

if __name__ == '__main__':

    # Choose pretrained model from task B
    state_dict_path = './outputs/task2/single_vgg_unfrozen_transoriginal_kaggleDRresized/model_single_vgg16_freezeFalse_transformoriginal_0.pth',
    # state_dict_path = './pretrained_DR_resize/pretrained/vgg16.pth',
    

    # Choose dataset
    dataset_name = 'DeepDRiD/'
    #dataset_name = 'kaggleDRresized/'

    # Choose to enable gradcam
    #gradcam = False
    gradcam = True


    # Choose preprocessing options
    preprocessing_options = {
        'ben_graham': True,
        'circle_crop': True,
        'sharpen': True,
        'clahe': True,
        #'gaussian_blur': 2,  # Radius of Gaussian blur
    }



    # The function runs training on all permutations of the parameters, and saves the metrics in a txt file
    # it calculates the average validation kappa score of each permutation across the number of loops specified.
    # freely remove processes you don't want to compare, for example transformcutoff, or channel.
    compare_training(
        dataset_name = dataset_name,
        state_dict_path = './outputs/task2/single_vgg_unfrozen_transoriginal_kaggleDRresized/model_single_vgg16_freezeFalse_transformoriginal_0.pth',
        attentions = ['none', 'self', 'channel', 'spatial'],
        subfolder_name='Compare_all',
        backbone_names = ["resnet18", "resnet34", "vgg16", "efficientnet_b0", "densenet121"],
        modes=['single', 'dual'],
        backbone_freezes=[False, True],
        train_transforms = {"transform": transform_train, "transformcutoff": transform_train_cutoff, "transformaggressive": transform_train_aggressive},
        loops=6,
        preprocessing_options=preprocessing_options
        )
    

    # Example usage for just running it on one combination, Get average of 6 runs. Without preprocessing
    compare_training(
        dataset_name = dataset_name,
        state_dict_path = './outputs/task2/single_vgg_unfrozen_transoriginal_kaggleDRresized/model_single_vgg16_freezeFalse_transformoriginal_0.pth',
        attentions = ['self'],
        subfolder_name='Average_kappa_for_one_model_combination',
        backbone_names = ["vgg16"],
        modes=['single'],
        backbone_freezes=[False],
        train_transforms = {"transformaggressive": transform_train_aggressive},
        loops=6,
        #preprocessing_options=preprocessing_options
        )
    

    # Just for gridsearching combinations of preprocesses
    grid_search_preprocessing()
    
    # validation preds for some ensemble methods
    val_pred_paths = [
        './outputs/task3/taskb/val_predictions_single_densenet121_freezeFalse_transformoriginal_none_4.csv',
        './outputs/task3/taskb/val_predictions_single_vgg16_freezeFalse_transformoriginal_none_0.csv',
        './outputs/task3/taskb/val_predictions_dual_resnet18_freezeFalse_transformoriginal_none_4.csv'
    ]

    # Compare ensemble methods
    scores = compare_ensembles(val_pred_paths)
    print("Cohen's kappa scores:", scores)
    
