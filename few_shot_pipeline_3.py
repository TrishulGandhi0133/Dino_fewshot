import os
import json
import time
import shutil
import random
from roboflow import Roboflow
from ultralytics import YOLO

def initialize_roboflow(api_key, project_name, work_name, dataset_version):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(work_name).project(project_name)
    dataset = project.version(dataset_version)
    return dataset

def download_dataset(dataset):
    print("Downloading the full dataset")
    dataset_dir = dataset.download("yolov8")
    dataset_path = dataset_dir.location
    return dataset_path

def sample_images(dataset_path, increment, sampled_images_set):
    # Directory paths
    images_dir = os.path.join(dataset_path, "train/images")
    labels_dir = os.path.join(dataset_path, "train/labels")

    # Create directories for the sampled dataset if they don't exist
    sampled_images_dir = os.path.join(dataset_path, "sampled/images")
    sampled_labels_dir = os.path.join(dataset_path, "sampled/labels")
    os.makedirs(sampled_images_dir, exist_ok=True)
    os.makedirs(sampled_labels_dir, exist_ok=True)

    # Get a list of all image files
    all_images = os.listdir(images_dir)
    remaining_images = list(set(all_images) - sampled_images_set)
    new_sampled_images = random.sample(remaining_images, increment)
    sampled_images_set.update(new_sampled_images)

    # Copy the sampled images and their corresponding labels
    for image in new_sampled_images:
        shutil.copy(os.path.join(images_dir, image), sampled_images_dir)
        label = image.replace('.jpg', '.txt')
        shutil.copy(os.path.join(labels_dir, label), sampled_labels_dir)

    return os.path.join(dataset_path, "sampled"), sampled_images_set

def create_data_config(sampled_train_path, validation_path):
    data_config_path = os.path.join(sampled_train_path, "data.yaml")
    data_config_content = f"""
names:
- aeroplane
- bicycle
- bird
- boat
- bottle
- bus
- car
- cat
- chair
- cow
- diningtable
- dog
- horse
- motorbike
- person
- pottedplant
- sheep
- sofa
- train
- tvmonitor
nc: 20
roboflow:
  license: CC BY 4.0
  project: pascal-voc-2012
  url: https://universe.roboflow.com/jacob-solawetz/pascal-voc-2012/dataset/1
  version: 1
  workspace: jacob-solawetz
test: ../test/images
train: {os.path.join(sampled_train_path, 'images')}
val: {os.path.join(validation_path, 'images')}
"""
    with open(data_config_path, 'w') as f:
        f.write(data_config_content)
    return data_config_path

def train_model(data_config_path, model_name):
    model = YOLO(f"{model_name}.pt")
    model.train(data=data_config_path, epochs=15, imgsz=640)  # Adjust epochs as needed
    return model

def evaluate_model(model):
    results = model.val()
    
    # Extract metrics from results_dict
    results_dict = results.results_dict
    map50 = results_dict['metrics/mAP50(B)']
    map95 = results_dict['metrics/mAP50-95(B)']
    
    return {'map50': map50, 'map95': map95}

def save_metrics(metrics, iteration, results):
    metrics.append({
        'iteration': iteration,
        'metrics': results
    })
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, default=str)

def print_metrics(metrics):
    for metric in metrics:
        print(f"Iteration: {metric['iteration']}, Metrics: {metric['metrics']}")

def few_shot_object_detection_pipeline(api_key, project_name, work_name, dataset_version, model_name, validation_path):
    dataset = initialize_roboflow(api_key, project_name, work_name, dataset_version)
    metrics = []

    dataset_path = download_dataset(dataset)

    #Clear old sample folders if it already exists
    sampled_images_dir = os.path.join(dataset_path, "sampled/images")
    sampled_labels_dir = os.path.join(dataset_path, "sampled/labels")

    if os.path.exists(sampled_images_dir):
        shutil.rmtree(sampled_images_dir)
    if os.path.exists(sampled_labels_dir):
        shutil.rmtree(sampled_labels_dir)
    
    sampled_images_set = set()
    num_images = 500
    iteration = 0
    sampled_train_path, sampled_images_set = sample_images(dataset_path, num_images, sampled_images_set)
    data_config_path = create_data_config(sampled_train_path, validation_path)

    while num_images <= 1500:
        print(f"Starting iteration with {num_images} images")
        model = train_model(data_config_path, model_name)
        results = evaluate_model(model)
        save_metrics(metrics, num_images, results)
        print(f"Completed iteration with {num_images} images")
        time.sleep(1)  # To avoid hitting API rate limits

        if num_images + 100 > 1500:
            # increment = 1500 - num_images
            break
        else:
            increment = 100
        num_images += increment
        sampled_train_path, sampled_images_set = sample_images(dataset_path, increment, sampled_images_set)

    print_metrics(metrics)
    print("Few-shot object detection pipeline completed.")

# Define parameters
api_key = "DqyBn9TcuvRKoMvycQyR"
work_name = "jacob-solawetz"
project_name = "pascal-voc-2012"
dataset_version = "1"
model_name = "yolov8x"  # Example: 'yolov8n', 'yolov8s', etc.
validation_path = "/home/srish/Documents/Pascal-VOC-2012-1/valid"  # Path to your predefined validation folder

# Run the pipeline
few_shot_object_detection_pipeline(api_key, project_name, work_name, dataset_version, model_name, validation_path)
