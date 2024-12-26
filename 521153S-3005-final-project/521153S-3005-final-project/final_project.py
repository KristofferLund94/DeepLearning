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

# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 20


class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

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


transform_train_original = transforms.Compose([
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

transform_train = transforms.Compose([
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
    transforms.RandomCrop((240, 240)),                      # Retain more information
    SLORandomPad((224, 224)),                               # Add padding to restore final size
    FundRandomRotate(prob=0.5, degree=15),                  # Smaller rotation for medical images
    transforms.RandomHorizontalFlip(p=0.5),                # Consider removing if orientation matters
    transforms.ColorJitter(brightness=(0.8, 1.2)),          # Subtle brightness variation
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    CutOut(mask_size=16, p=0.5)                             # Add CutOut 
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
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

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./Downloads/DeepLearning/521153S-3005-final-project/521153S-3005-final-project/test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

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


class MyModel(nn.Module):
    def __init__(self, backbone_name="resnet18", freeze_backbone=False, num_classes=5, dropout_rate=0.5):
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
            feature_dim = 25088  # Output size of VGG-16's features block
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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        

    def forward(self, x):
        x = self.backbone(x)  # Pass through the backbone
        x = self.fc(x)        # Pass through the custom classifier
        return x


class MyDualModel(nn.Module):
    def __init__(self, backbone_name="resnet18", freeze_backbone=False, num_classes=5, dropout_rate=0.5):
        super().__init__()

        # Select the backbone
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove classification layer

        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=True)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "vgg16":
            backbone = models.vgg16(pretrained=True)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()  # Remove classification layer
            
        elif backbone_name == "densenet121":
            backbone = models.densenet121(pretrained=True)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()  # Remove classification layer
            
        else:
            raise ValueError(f"Backbone '{backbone_name}' is not supported.")


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

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

def compare_training(dataset_name = 'DeepDRiD/', subfolder_name='default_output', backbone_names=["vgg16"], modes=['single'], backbone_freezes=[False], train_transforms={"transformoriginal": transform_train_original}, loops=6):
    mainfolder_path = './'
    outfolder_path = 'outputs/task2/'
    subfolder_path = mainfolder_path + outfolder_path + subfolder_name
    dataset_path = mainfolder_path + dataset_name
    log_file_path = subfolder_path + '/metrics_log.txt'
    os.makedirs(subfolder_path, exist_ok=True)
    

    val_metrics_dictionary = {}
    kappa_averages = {}

    for mode in modes:
        for backbone_name in backbone_names:
            for freeze_backbone in backbone_freezes:
                for train_transform in train_transforms:
                    kappa_scores = []
                    for loop in range(loops):
                        model_name = mode + '_' + backbone_name + '_freeze' + str(freeze_backbone) + '_' + train_transform
                        loop_name = model_name + '_' + str(loop)
                        model_path = subfolder_path + '/model_' + loop_name + '.pth'
                        prediction_path = subfolder_path + '/test_predictions_' + loop_name + '.csv'

                        # Choose between 'single image' and 'dual images' pipeline
                        # This will affect the model definition, dataset pipeline, training and evaluation

                        # mode = 'single'  # forward single image to the model each time
                        # mode = 'dual'  # forward two images of the same eye to the model and fuse the features

                        assert mode in ('single', 'dual')

                        # Define the model
                        if mode == 'single':
                            model = MyModel(backbone_name=backbone_name, freeze_backbone=freeze_backbone)
                        else:
                            model = MyDualModel()

                        print(model, '\n')
                        print('Pipeline Mode:', mode)

                        # Create datasets
                        train_dataset = RetinopathyDataset(dataset_path + 'train.csv', dataset_path + 'train/', train_transforms[train_transform], mode)
                        val_dataset = RetinopathyDataset(dataset_path + 'val.csv', dataset_path + 'val/', transform_test, mode)
                        test_dataset = RetinopathyDataset(dataset_path + 'test.csv', dataset_path + 'test/', transform_test, mode, test=True)

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
                        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

                        # Train and evaluate the model with the training and validation set
                        model = train_model(
                            model, train_loader, val_loader, device, criterion, optimizer,
                            lr_scheduler=lr_scheduler, num_epochs=num_epochs,
                            checkpoint_path = model_path
                        )

                        # Load the pretrained checkpoint
                        state_dict = torch.load(model_path, map_location='cuda', weights_only=True)
                        model.load_state_dict(state_dict, strict=True)

                        # Make predictions on testing set and save the prediction results
                        evaluate_model(model, test_loader, device, test_only=True, prediction_path=prediction_path)

                        
                        val_metrics = evaluate_model(model, val_loader, device)
                        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
                        val_metrics_dictionary[loop_name] = f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f} Recall: {val_recall:.4f}\n'

                        kappa_scores.append(val_kappa)
            
                    kappa_averages[model_name] = np.mean(kappa_scores)
                    with open(log_file_path, 'w') as f:
                        print(f'All Metrics:\n{val_metrics_dictionary}\n')
                        print(f'All Metrics:\n{val_metrics_dictionary}\n', file=f)
                        
                        print(f'Average Kappa Scores:\n{kappa_averages}\n')
                        print(f'Average Kappa Scores:\n{kappa_averages}\n', file=f)
    



if __name__ == '__main__':

    #backbone_names = ["resnet18", "resnet34", "vgg16", "efficientnet_b0", "densenet121"]
    #modes = ["single", "dual"]
    #backbone_freezes = [False, True]
    #train_transforms = {"transform": transform_train, "transformcutoff": transform_train_cutoff, "transformoriginal": transform_train_original}

    #dataset_name = 'DeepDRiD/'
    dataset_name = 'kaggleDRresized/'

    compare_training(
        dataset_name = dataset_name,
        subfolder_name='single_vgg_unfrozen_transoriginal_kaggleDRresized',
        backbone_names=["vgg16"],
        modes=['single'],
        backbone_freezes=[False],
        train_transforms={"transform": transform_train, "transformoriginal": transform_train_original},
        loops=1)
    
