# Enhancing-Object-Detection-for-Autonomous-Driving-using
Enhancing object detection
This Jupyter Notebook implements and trains two different object detection models, **YOLOv8** (an ultra-fast deep learning model) and **Faster R-CNN** (a popular two-stage model), on a self-driving car dataset.

The overall pipeline is: **Install Dependencies** → **Download Dataset** → **Prepare Data** → **Train/Evaluate YOLOv8** → **Train/Evaluate Faster R-CNN** (an alternative approach).

***

## 1. Setup and Data Preparation

This phase prepares the environment and structures the raw data for model training.

### 1. Install Dependencies
The first step uses the `!pip install` command to set up the necessary Python libraries for deep learning, data handling, and visualization, including:
* **`ultralytics`**: The framework for using the **YOLOv8** object detection model.
* **`roboflow`**: A platform used to download the custom self-driving car dataset.
* **`torch` / `torchvision`**: The core libraries for building and training deep learning models like **Faster R-CNN** (used later).

### 2. Download Dataset from Roboflow
The code uses the `Roboflow` API key to programmatically access and download a dataset named **"Self-Driving-Car" (version 3)**. The dataset is downloaded in the **YOLOv8 format**, which means images and their corresponding bounding box annotations (labels) are saved as separate `.jpg` and `.txt` files.

### 3. Exploratory Data Analysis (EDA)
The script analyzes the downloaded label files to determine the **class distribution** of the objects to be detected. This showed 11 distinct classes, with `Class 1` ('car') having the highest count (127,873 instances). It also displays a sample image to visually confirm the data .

### 4. Dataset Preparation: Train/Validation/Test Split
The code splits the full dataset into three standard subsets for machine learning:
* **Training Set**: 70% of the images are used to train the model.
* **Validation Set**: 20% of the images are used for tuning hyperparameters and evaluating the model during training.
* **Test Set**: 10% of the images are used for final, unbiased evaluation.

The script creates separate folders (`train/images`, `train/labels`, `valid/images`, `valid/labels`, etc.) and copies the corresponding image and label files into the correct directories for YOLOv8 to consume.

### 5. Creating `data.yaml`
A YAML file named `data.yaml` is created. This configuration file is essential for YOLOv8 training, as it contains all metadata the model needs, including:
* **Number of classes (`nc`):** 11
* **Class names (`names`):** e.g., 'car', 'pedestrian', 'trafficLight-Green'.
* **Paths** to the `train`, `val`, and `test` image folders.

***

## 6. YOLOv8 Model Training and Evaluation

This section focuses on using the specialized YOLOv8 model.

### 6. Train YOLOv8
The code initializes the YOLOv8-nano model (`yolov8n.pt`), which is the fastest and smallest version, suitable for quick training. It then trains the model using the following key settings:
* **`data`**: Specifies the `data.yaml` file.
* **`epochs`**: 5 (a short training run).
* **`imgsz`**: 320 (image size for training).
* **`batch`**: 4 (small batch size for fast iteration).
* **`optimizer`**: SGD.

### 7. YOLO Training Results (Analysis)
The training logs and metrics are loaded from the generated `results.csv` file.

The Mean Average Precision (**mAP**) scores over the 5 epochs show improvement:
* **Final mAP@50 (General performance)**: **0.337**
* **Final mAP@50-95 (Performance across various Intersection over Union thresholds)**: **0.198**

The training process shows that all **losses** (`box_loss`, `cls_loss`, `dfl_loss`) continuously **decrease** over the 5 epochs . This indicates that the model is learning from the training data.

***

## 8. Faster R-CNN Implementation (Alternative Approach)

This advanced section prepares the data and model for an alternative object detection architecture: **Faster R-CNN** using PyTorch/Torchvision.

### Custom Data Loading
A custom class, `YOLODataset`, is defined to load the YOLO-formatted annotations and images, converting the normalized coordinates into pixel bounding box coordinates (`x1`, `y1`, `x2`, `y2`) required by Faster R-CNN.

### Model Setup
The `fasterrcnn_resnet50_fpn` model is initialized with pre-trained weights, and the final layer (`box_predictor`) is replaced with a new layer adapted for the **11 custom classes** plus one background class, totaling `num_classes = 12`. The model is moved to the GPU (`cuda`) for efficient training.

### Training Loop
A standard PyTorch training loop is implemented with:
* **Epochs**: 5.
* **Optimization**: `Adam` optimizer.
* **Mixed Precision**: Uses `torch.amp.autocast` and `torch.amp.GradScaler` to leverage **Automatic Mixed Precision (AMP)** for faster training on the GPU.
* The loop tracks and prints the **Training Loss** and **Validation Loss** for each epoch, showing that both losses generally decrease .

***

## 9. Inference on a New Image

The final section demonstrates how to use the *trained YOLOv8 model* (`best.pt`) to make predictions on a new, uploaded image. The image is loaded, passed to the model for prediction (`model.predict()`), and the resulting image with bounding box annotations is displayed .
