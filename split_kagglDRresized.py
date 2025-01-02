import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
csv_file = "kaggleDRresized/trainLabels_cropped.csv"
image_folder = "kaggleDRresized/resized_train_cropped"
output_folder = "kaggleDRresized"

# Create output folders
os.makedirs(output_folder, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

# Read CSV
df = pd.read_csv(csv_file)

# Extract patient number and left/right info
df['PatientNumber'] = df.iloc[:, 2].str.split('_').str[0]
df['Eye'] = df.iloc[:, 2].str.split('_').str[1]

# Calculate patient-level max DR level
patient_max_DR = df.groupby('PatientNumber')[df.columns[3]].max()
df['PatientMaxDR'] = df['PatientNumber'].map(patient_max_DR)

# Extract the left eye's target for stratification
left_eye_targets = df[df['Eye'] == 'left'][['PatientNumber', df.columns[3]]]
left_eye_targets = left_eye_targets.rename(columns={df.columns[3]: 'Target'})

# Group by patient number
patient_groups = df.groupby('PatientNumber')

# Split patients into train, validation, and test sets, stratified by left eye's target
train_patients, test_patients = train_test_split(
    left_eye_targets['PatientNumber'],
    test_size=0.15,
    random_state=42,
    stratify=left_eye_targets['Target']
)
train_patients, val_patients = train_test_split(
    train_patients,
    test_size=0.1765,  # 0.1765 x 0.85 ≈ 0.15
    random_state=42,
    stratify=left_eye_targets[left_eye_targets['PatientNumber'].isin(train_patients)]['Target']
)

# Helper function to organize splits and generate metadata
def organize_split(patients, split):
    split_data = []
    split_folder = os.path.join(output_folder, split)
    for patient in patients:
        patient_folder = os.path.join(split_folder, f"{patient}")
        os.makedirs(patient_folder, exist_ok=True)
        
        patient_data = patient_groups.get_group(patient)
        patient_DR_Level = patient_max_DR[patient]
        
        for _, row in patient_data.iterrows():
            eye = row['Eye']
            target = row.iloc[3]
            image_id = f"{patient}_{'l' if eye == 'left' else 'r'}1"
            img_path = os.path.join(f"{patient}", f"{image_id}.jpeg")
            src_path = os.path.join(image_folder, f"{row.iloc[2]}.jpeg")
            dst_path = os.path.join(patient_folder, f"{image_id}.jpeg")
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                split_data.append({
                    "patient_id": patient,
                    "image_id": image_id,
                    "img_path": img_path,
                    "left_eye_DR_Level": target if eye == "left" else "",
                    "right_eye_DR_Level": target if eye == "right" else "",
                    "patient_DR_Level": patient_DR_Level
                })
            else:
                print(f"Image not found: {src_path}")
    
    # Save split CSV
    csv_path = os.path.join(output_folder, f"{split}.csv")
    pd.DataFrame(split_data).to_csv(csv_path, index=False)
    print(f"{split.capitalize()} metadata saved to: {csv_path}")

# Organize files and create CSVs for each split
organize_split(train_patients, "train")
organize_split(val_patients, "val")
organize_split(test_patients, "test")

print("Dataset split and organization completed!")