import os
import pickle
import time
import socket
import os
import logging
import torch
import sys
from pathlib import Path
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from PIL import Image # type: ignore
from tqdm import tqdm
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataloader import DataManager
from src.experiment_design.utils import DataUtils

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

dataset_path = 'data/onion/testing'
yaml_file_path = 'config/model_config.yaml'
weight_path = 'data/runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OnionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()
        if self.transform:
            image = self.transform(image)
        return image, original_image, self.image_files[idx]


# Custom dataset transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Load your dataset
dataset = OnionDataset(root=dataset_path, transform=transform)

# Custom collate_fn to avoid batching the PIL Image
def custom_collate_fn(batch):
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# ------------------ FIX THIS PART (START) ------------------
# Initialize the YOLO model
model = WrappedModel(config=yaml_file_path)

# Load the model
model.eval()
# ------------------ FIX THIS PART (END) ------------------



print(device)
server_address = ('10.0.0.245', 12345)  # Update with your server's address
print(server_address)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)


def test_split_performance(client_socket, split_layer_index):
    correct = 0
    total = 0
    start_time = time.time()
    # Lists to store times
    host_times = []
    travel_times = []
    server_times = []

    with torch.no_grad():
        for input_tensor, original_image, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer_index}"):
            # Measure host processing time
            input_tensor = input_tensor.to(model.device)
            host_start_time = time.time()
            # Processing with the model...
            out = model(input_tensor, end=split_layer_index)
            data_to_send = (out,original_image[0].size)
            compressed_output,compressed_size = DataUtils.compress_data(data_to_send)
            host_end_time = time.time()
            host_times.append(host_end_time - host_start_time)

            # Send data to server
            travel_start_time = time.time()
            client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
            client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
            client_socket.sendall(compressed_output)

            # Receive and unpack server response
            data = client_socket.recv(4096)
            prediction, server_processing_time = pickle.loads(data)
            travel_end_time = time.time()
            travel_times.append(travel_end_time - travel_start_time)  # This includes server time
            print(prediction)
            server_times.append(server_processing_time)  # Log server time for each image

    end_time = time.time()
    processing_time = end_time - start_time
    # accuracy = 100 * correct / total
    # print(f'Accuracy of the model on the test images: {accuracy} %')
    # print(f"Compressed Size in bytes: {compressed_size}")
    # Calculate average times for each part
    total_host_time = sum(host_times)
    total_travel_time = sum(travel_times) - sum(server_times)   # Correcting travel time
    total_server_time = sum(server_times)
    total_processing_time = total_host_time + total_travel_time + total_server_time
    print(f"Total Host Time: {total_host_time:.2f} s, Total Travel Time: {total_travel_time:.2f} s, Total Server Time: {total_server_time:.2f} s")

    return (total_host_time,total_travel_time,total_server_time,total_processing_time)



total_layers = 23 # len(list(model.backbone.body.children()))
print(total_layers)
time_taken = []

for split_layer_index in range(1, total_layers):  # Assuming layer 0 is not a viable split point
    host_time,travel_time,server_time,processing_time = test_split_performance(client_socket, split_layer_index)
    print(f"Split at layer {split_layer_index}, Processing Time: {processing_time:.2f} seconds")
    time_taken.append((split_layer_index, host_time, travel_time, server_time, processing_time))

best_split, host_time, travel_time, server_time, min_time = min(time_taken, key=lambda x: x[4])
print(f"Best split at layer {best_split} with time {min_time:.2f} seconds")
for i in time_taken:
    split_layer_index, host_time, travel_time, server_time, processing_time = i
    print(f"Layer {split_layer_index}: Host Time = {host_time}, Travel Time = {travel_time}, Server Time = {server_time}, Total Processing Time = {processing_time}")

df = pd.DataFrame(time_taken, columns=["Split Layer Index", "Host Time", "Travel Time", "Server Time", "Total Processing Time"])

# Save the DataFrame to an Excel file
df.to_excel("split_layer_times.xlsx", index=False)

print("Data saved to split_layer_times.xlsx")
# split_layer_index = 3
# host_time,travel_time,server_time,processing_time = test_split_performance(client_socket, split_layer_index)
client_socket.close()