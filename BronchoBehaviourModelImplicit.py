# Standard library imports
import argparse
import csv
import os
import random
import shutil
import time

# Third-party imports for data handling and transformations
from PIL import Image
import numpy as np

# PyTorch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
import torchvision.transforms as T

# Imports for model evaluation
from sklearn.metrics import precision_score, recall_score, f1_score

# Visualization imports
import matplotlib.pyplot as plt

# Local module imports
from DataHandling.Episode import Episode, EpisodeManager


#Other
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
#from torch.cuda.amp import autocast
import json
from collections import defaultdict

# Action to name mapping
actionToName = {
    -1: "No Input",
    0: "Down",
    1: "Up",
    2: "Right",
    3: "Left",
    4: "Forward",
    5: "Backward",
}

# Custom Dataset class
class BronchosopyDataset(Dataset):
    def __init__(self, databasePath, transform=None, blurSigma=0, maxEpisodes=100, shuffle = False):
        self.episodeManager = EpisodeManager(mode = "read", loadLocation = databasePath)


        self.episodes, self.lenght, self.episodeFrameIndexStart = self.episodeManager.loadAllEpisodes(cacheImages=False, maxEpisodes=maxEpisodes, shuffle=shuffle, shuffleSeed=1)

        if blurSigma > 0:
            for i, episode in enumerate(self.episodes):
                pass
                episode.apply_gaussian_blur_with_normalization(blurSigma)

        self.transform = transform

        self.actionToIndex = {"u": 0, "d": 1, "l": 2, "r": 3, "f": 4, "b": 5}


        self.cache = {}
        

    def __len__(self):
        return self.lenght


    def getEpisodeAndSubIndex(self, index):
        episodeIndex = 0
        subIndex = index



        for i in range(len(self.episodeFrameIndexStart)-1, 0, -1):
            if index >= self.episodeFrameIndexStart[i]:
                episodeIndex = i
                subIndex = index - self.episodeFrameIndexStart[i]
                break

        return episodeIndex, subIndex


    def __getitem__(self, idx):
        
        if idx in self.cache.keys():
            return self.cache[idx]
        else:
            episodeIndex, subIndex = self.getEpisodeAndSubIndex(idx)

            #print(f"Getting index {idx} episodeIndex {episodeIndex} subIndex {subIndex} using frameStart {self.episodeFrameIndexStart}")


            episodeAndIndex = (episodeIndex, subIndex)


            frame = self.episodes[episodeIndex][subIndex]

            image = Image.fromarray(frame.image)

            if self.transform:
                image = self.transform(image)


            data = frame.data
            state = [frame.state["bendReal_deg"], frame.state["rotationReal_deg"], frame.state["extensionReal_mm"]]

            goal = data["pathId"]

            paths = data["paths"]

            goalPath = paths[str(goal)]

            goalBbox = goalPath["bbox"]

            "bbox to center and size"
            goalCenter = [(goalBbox[0] + goalBbox[2]) / 2, (goalBbox[1] + goalBbox[3]) / 2]
            #normalize, so center of image is 0,0 image size is 400,400
            goalCenter[0] = (goalCenter[0] - 200)/200
            goalCenter[1] = (goalCenter[1] - 200)/200


            goalSize = [goalBbox[2] - goalBbox[0], goalBbox[3] - goalBbox[1]]
            #normalize, so image size is 400,400
            goalSize[0] = goalSize[0] / 200
            goalSize[1] = goalSize[1] / 200

            stateInput = torch.tensor(state, dtype=torch.float)
            #goalInput = torch.tensor(goalCenter + goalSize, dtype=torch.float)
            goalInput = torch.tensor(goalCenter, dtype=torch.float)

            action = frame.action["char_value"] # char value u, d, l, r, f, b or none
            #print("")
            #print(frame.action)

            action = frame.action["model_value"] # char value u, d, l, r, f, b or none
            #print("")
            #print(frame.action)

            #convert list of float directly to tensor
            action = torch.tensor(action, dtype=torch.float)


            

            if torch.sum(action) == 0:
                action = torch.zeros(6)
            else:
                action = action / torch.sum(action)
                
            

            self.cache[idx] = (image, stateInput, goalInput, action, episodeAndIndex)
            
            return image, stateInput, goalInput, action, episodeAndIndex


class FilteredUpsampledDataset(Dataset):
    def __init__(self, originalDataset, doUpsample=True, maxOversamlingFactor=12, upsampleIndex=0):

        print("Filtering and upsampling dataset")
        self.upsampleIndex = upsampleIndex
        self.primaryDataset = originalDataset

        self.maxOversamlingFactor = maxOversamlingFactor

        validIndices = self.filterInvalidIndices()


        if(doUpsample):
            classCounts = self.countClasses(validIndices)

            oversamplingFactors = self.calculateOversamplingFactors(classCounts)

            oversampledIndices = self.oversampleIndices(validIndices, oversamplingFactors)

            self.oversampledIndices = oversampledIndices
        else:
            self.oversampledIndices = validIndices

    def filterInvalidIndices(self):
        validIndices = []

        for i in range(len(self.primaryDataset)):
            features, state, goal, label, _ = self.primaryDataset[i]

            #label is one-hot encoded, invalid if all zeros
            if torch.sum(label) > 0:
                validIndices.append(i)

            print(f"\rFiltering invalid indices {i}/{len(self.primaryDataset)} - valid: {len(validIndices)}, invalid: {i - len(validIndices)}, label: {label}                      ", end="")
            #time.sleep(0.01)
        print("")


        return validIndices
    
    def countClasses(self, validIndices):
        classCounts = torch.zeros(6)

        for i, idx in enumerate(validIndices):
            features, state, goal, label, _ = self.primaryDataset[idx]

            classCounts += label

            print(f"\rCounting classes {i}/{len(validIndices)}", end="")

        print("")
        print(f"Class Counts: {classCounts}")
        return classCounts
    
    def calculateOversamplingFactors(self, classCounts):
        oversamplingFactors = {}

        #find factor such that each class has equal amount of samples

        #maxCount = max(classCounts.values())

        #second largest count

        sortedCounts = torch.sort(classCounts, descending=True).values

        baseCount = sortedCounts[self.upsampleIndex]
        
        factor=baseCount/classCounts

        oversamplingFactors = torch.clamp(factor, min = 0.1, max=self.maxOversamlingFactor)

        print(f"Sampling factors {oversamplingFactors}")


        #for classId in classCounts:
        #    count = classCounts[classId]
        #    factor = baseCount / count
#
        #    if factor > self.maxOversamlingFactor:
        #        factor = self.maxOversamlingFactor
#
        #    oversamplingFactors[classId] = factor
        #
        #print("Sampling factors")
        #for classId in oversamplingFactors:
        #    print(f"Class {classId} factor: {oversamplingFactors[classId]}")
#
        return oversamplingFactors
    
    def oversampleIndices(self, validIndices, oversamplingFactors):
        oversampledIndices = []

        accumulator = torch.zeros(6)
        counts = torch.zeros(6)

        for idx in validIndices:
            features, state, goal, label, _ = self.primaryDataset[idx]

            probablities = label

            factors = probablities * oversamplingFactors

            toAdd = factors + accumulator


            #while any of the values of toAdd is greater than probability
            while torch.all(toAdd >= probablities):

                #print(f"\rAdding sample with probabilties: {probablities}, leftToAdd: {toAdd}        ", end="")

                toAdd -= probablities

                counts += probablities

                oversampledIndices.append(idx)


            accumulator = toAdd #remainder





            #classId = torch.argmax(label).item()
#
            #factor = oversamplingFactors[classId]
#
            #toAdd = factor+accumulator[classId]
#
            #toAddActual = int(toAdd)
            #remainder = toAdd - toAddActual
            #accumulator[classId] = remainder
#
            #oversampledIndices.extend([idx] * toAddActual)
            #counts[classId] += toAddActual
#
            print(f"\rOversampling indices {idx}/{len(validIndices)}, Oversampled {len(oversampledIndices)}          ", end="")
        print("")
        print("Counts After Resampling")
        
        print(counts)

        return oversampledIndices
    
    def __len__(self):
        return len(self.oversampledIndices)
    
    def __getitem__(self, idx):
        features, state, goal, label, episodeAndIndex = self.primaryDataset[self.oversampledIndices[idx]]
        #label to index
        #classId = torch.argmax(label).item()
        return features, state, goal, label, episodeAndIndex





# Define noise-adding functions
def add_gaussian_noise_01(x):
    """Adds Gaussian noise with a noise level of 0.01."""
    return add_gaussian_noise(x, 0.01)

def add_gaussian_noise_03(x):
    """Adds Gaussian noise with a noise level of 0.03."""
    return add_gaussian_noise(x, 0.03)

def add_gaussian_noise(x, noise_level):
    """General function to add Gaussian noise to the input tensor."""
    return x + torch.randn_like(x) * noise_level

class DataAugmentationDataset(Dataset):

    

    def __init__(self, originalDataset, grayscale=True, augmentation_factor=8, doStatePerturbation=True, doGoalPerturbation=True):
        """
        :param originalDataset: The original dataset to augment
        :param grayscale: Boolean, True if the dataset images are grayscale
        :param augmentation_factor: Integer, how many augmentations to apply per image
        """
        self.doStatePerturbation = doStatePerturbation
        self.doGoalPerturbation = doGoalPerturbation
        self.primaryDataset = originalDataset
        self.grayscale = grayscale
        self.augmentation_factor = augmentation_factor
        
        # Define possible transformations for grayscale images
        if self.grayscale:
            self.image_transform_options = [
                T.ColorJitter(brightness=0.1, contrast=0.1),  # Brightness and contrast for grayscale
                T.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness and contrast for grayscale
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Gaussian Blur
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translation
                #T.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # Random crop
                T.Lambda(add_gaussian_noise_01),  # Add Gaussian noise 0.01
                T.Lambda(add_gaussian_noise_03)  # Add Gaussian noise 0.03
            ]
        else:
            # If working with RGB images, allow for more advanced augmentations
            self.image_transform_options = [
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Full ColorJitter
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Full ColorJitter
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Gaussian Blur
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translation
                #T.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # Random crop
                T.Lambda(add_gaussian_noise_01),  # Add Gaussian noise 0.01
                T.Lambda(add_gaussian_noise_03)  # Add Gaussian noise 0.03
            ]

    def __len__(self):
        # The length of the dataset is the number of samples in the original dataset multiplied by the augmentation factor
        return len(self.primaryDataset) * self.augmentation_factor
    
    def __getitem__(self, idx):
        # Calculate the original index and transform to apply
        actualIdx = idx // self.augmentation_factor
        transformIdx = idx % self.augmentation_factor
        
        # Fetch the image, state, goal, label from the original dataset
        image, state, goal, label, episodeAndIndex = self.primaryDataset[actualIdx]

        
        # Apply random transformations if transformIdx is not 0
        if transformIdx != 0:

            # Convert image to PIL Image if it's a tensor
            #if isinstance(image, torch.Tensor):
            #    image = T.ToPILImage()(image)



            image = self.apply_random_transforms(image)

            # Convert image back to tensor
            #image = T.ToTensor()(image)

            # Apply state and goal perturbations
            if self.doStatePerturbation:
                state = self.perturb_state(state, transformIdx)
            if self.doGoalPerturbation:
                goal = self.perturb_goal(goal, transformIdx)
        
        return image, state, goal, label, episodeAndIndex
    
    def apply_random_transforms(self, image):
        """
        Apply 1 to 3 random image transformations.
        If the image is grayscale, only brightness and contrast are adjusted.
        If the image is RGB, full color jitter (hue, saturation) is applied.
        """
        num_transforms = random.randint(1, 3)  # Choose between 1 and 3 transformations
        selected_transforms = random.sample(self.image_transform_options, num_transforms)
        
        # Compose and apply the selected transforms
        composed_transform = T.Compose(selected_transforms)

        # If the image is grayscale but you're using ColorJitter, convert it to RGB first
        if self.grayscale:
            image = self.convert_grayscale_to_rgb(image)
        
        transformed_image = composed_transform(image)
        
        # Convert back to grayscale if needed
        if self.grayscale:
            transformed_image = T.Grayscale()(transformed_image)
        
        return transformed_image
    
    def convert_grayscale_to_rgb(self, image):
        """
        Convert a grayscale image to an RGB image so ColorJitter can be applied.
        """
        #return image.convert("RGB")
    
    def perturb_state(self, state, transformIdx):
        """
        Apply small perturbations to the state relative to its magnitude.
        """
        if transformIdx == 0:
            return state  # Original state, no perturbation

        # Apply noise proportional to the state values
        noise_factor = 0.05  # You can adjust the noise factor
        perturbed_state = state + noise_factor * (torch.randn_like(state)-0.5) * torch.abs(state)

        return perturbed_state
    
    def perturb_goal(self, goal, transformIdx):
        """
        Apply small perturbations to the goal relative to its magnitude.
        """
        if transformIdx == 0:
            return goal  # Original goal, no perturbation

        # Apply noise proportional to the goal values
        goal_perturbation_factor = 0.03  # Smaller perturbation compared to state
        perturbed_goal = goal + goal_perturbation_factor * (torch.randn_like(goal)-0.5) * torch.abs(goal)

        return perturbed_goal


# Multi-input CNN model
class MultiInputModel(nn.Module):
    def __init__(self, numStates=3, numGoal=4, imageSize=50, channels=1, classes=6):
        super(MultiInputModel, self).__init__()
        
        # Convolutional layers for the image input
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers for the additional input
        def conv2d_output_size(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        conv1_size = conv2d_output_size(imageSize)
        pool1_size = conv1_size // 2
        conv2_size = conv2d_output_size(pool1_size)
        pool2_size = conv2_size // 2
        conv3_size = conv2d_output_size(pool2_size)
        pool3_size = conv3_size // 2

        flattened_size = 32 * pool3_size * pool3_size

        self.fc1_state = nn.Linear(in_features=numStates, out_features=32)
        self.fc1_goal = nn.Linear(in_features=numGoal, out_features=32)

        concatenated_size = 32 + 32 + flattened_size
        
        # Fully connected layers for the combined output
        #self.fc1_combined = nn.Linear(in_features=concatenated_size, out_features=256)
        self.fc2_combined = nn.Linear(in_features=concatenated_size, out_features=128)
        self.fc3_combined = nn.Linear(in_features=128, out_features=classes)



        

    def forward(self, image, state, goal):
        # CNN pathway for image input
        x1 = self.pool(F.relu(self.conv1(image)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))

        state = F.relu(self.fc1_state(state))
        goal = F.relu(self.fc1_goal(goal))

        x = torch.cat((x1, goal, state), dim=1)

        #x = F.relu(self.fc1_combined(x))
        x = F.relu(self.fc2_combined(x))
        x = self.fc3_combined(x)
        
        return x

# Create a folder if it doesn't exist, and add a number if it does
def create_output_folder(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path
    else:
        i = 1
        new_path = f"{base_path}_{i}"
        while os.path.exists(new_path):
            i += 1
            new_path = f"{base_path}_{i}"
        os.makedirs(new_path)
        return new_path

def save_metrics_to_csv(epoch, train_metrics, random_val_metrics, seq_val_metrics, file_path):
    # Check if the file exists to determine if headers need to be written
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Training Precision', 'Training Recall', 'Training F1',
                             'Random Val Loss', 'Random Val Accuracy', 'Random Val Precision', 'Random Val Recall', 'Random Val F1',
                             'Sequence Val Loss', 'Sequence Val Accuracy', 'Sequence Val Precision', 'Sequence Val Recall', 'Sequence Val F1'])
        
        # Write the metrics for the current epoch
        writer.writerow([
            epoch,
            train_metrics['loss'], train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1'],
            random_val_metrics['loss'], random_val_metrics['accuracy'], random_val_metrics['precision'], random_val_metrics['recall'], random_val_metrics['f1'],
            seq_val_metrics['loss'], seq_val_metrics['accuracy'], seq_val_metrics['precision'], seq_val_metrics['recall'], seq_val_metrics['f1']
        ])

# Plot loss and save it
def plot_and_save_loss_curve(train_loss_values, random_val_loss_values, seq_val_loss_values, output_folder):
    plt.figure()
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(random_val_loss_values, label='Random Validation Loss')
    plt.plot(seq_val_loss_values, label='Sequence Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_curve.png'))
    plt.close()




class BronchoBehaviourModelImplicit:
    def __init__(self, model_path, device='cuda'):
        # Initialize the model and load the weights
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = MultiInputModel(numStates=3, numGoal=2, imageSize=40, channels=3, classes=6)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        

        self.transform = T.Compose([
            #transforms.Grayscale(),                     # Convert images to grayscale
            T.Resize((40, 40)),                # Resize images to 50x50 pixels
            T.ToTensor(),                      # Convert images to PyTorch tensors
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         # Normalize images
        ])


        # Action to index mapping
        self.action_to_index = {"u": 0, "d": 1, "l": 2, "r": 3, "f": 4, "b": 5}

    def predict(self, image, state, paths, selected_path_key):
        #print(image)
        # 1. Transform the image
        image_tensor = self._transform_image(image)

        # 2. Extract the relevant state information (bend, rotation, extension)

        state = [state["bendReal_deg"], state["rotationReal_deg"], state["extensionReal_mm"]]
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        # 3. Get the goal from the selected path
        goal_tensor = self._extract_goal_from_path(paths, selected_path_key)

        # 4. Feed the inputs to the model to get action probabilities
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        goal_tensor = goal_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor, state_tensor.unsqueeze(0), goal_tensor.unsqueeze(0))

        probabilities = F.softmax(logits, dim=1)

        # 5. Sample an action based on the probabilities
        action_index = self._sample_action(probabilities)

        return action_index

    def _transform_image(self, image):
        """Applies the necessary image transformations."""
        if isinstance(image, Image.Image):
            return self.transform(image)
        else:
            # Convert to PIL Image if it's a numpy array
            image = Image.fromarray(image)
            return self.transform(image)

    def _extract_goal_from_path(self, paths, selected_path_key):
        """Extracts the goal from the path dictionary using the selected path key."""
        selected_path = paths[selected_path_key]
        goal_bbox = selected_path.bbox

        # Convert bbox to center and size, and normalize based on image size (400x400)
        goal_center = [(goal_bbox[0] + goal_bbox[2]) / 2, (goal_bbox[1] + goal_bbox[3]) / 2]
        goal_center[0] = (goal_center[0] - 200) / 200  # Normalize x
        goal_center[1] = (goal_center[1] - 200) / 200  # Normalize y

        goal_size = [goal_bbox[2] - goal_bbox[0], goal_bbox[3] - goal_bbox[1]]
        goal_size[0] = goal_size[0] / 200  # Normalize width
        goal_size[1] = goal_size[1] / 200  # Normalize height

        # Combine the center and size as the goal tensor
        #goal_tensor = torch.tensor(goal_center + goal_size, dtype=torch.float32)
        goal_tensor = torch.tensor(goal_center, dtype=torch.float32)
        return goal_tensor

    def _sample_action(self, probabilities):
        """Sample an action from the output probabilities."""
        probabilities = probabilities.squeeze().cpu().numpy()  # Convert to numpy array
        action_index = random.choices(range(len(probabilities)), probabilities)[0]

        #print out the probabilities
        actionToName = {
            -1: "No Input",
            0: "D",
            1: "U",
            2: "R",
            3: "L",
            4: "F",
            5: "B"
        }

        probabilityString = ", ".join([f"{actionToName[i]}: {probabilities[i]:.2f}" for i in range(len(probabilities))])

        print(f"Action Probabilities: {probabilityString}, Selected Action: {actionToName[action_index]} with probability {probabilities[action_index]:.2f}")

        

        return action_index



# Main function to handle training
def main(epochs = 50, learningRate = 0.0001, modelSavePath = "modelImplicit.pth"):
    # Image transformation
    transform = T.Compose([
        #transforms.Grayscale(),                     # Convert images to grayscale
        T.Resize((40, 40)),                # Resize images to 50x50 pixels
        T.ToTensor(),                      # Convert images to PyTorch tensors
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         # Normalize images
    ])

    # Load your dataset
    # Assuming image_paths, extra_features, and labels are preloaded lists
    # Example: Replace these with your actual data loading logic

    output_folder = create_output_folder("runs/implictTraining")

    # File to save metrics
    metrics_file_path = os.path.join(output_folder, "metrics.csv")

    # Create the custom dataset and DataLoader
    dataset = BronchosopyDataset("DatabaseLabelled", transform=transform, blurSigma=0, maxEpisodes=100, shuffle=False)

    dataset = FilteredUpsampledDataset(dataset, doUpsample=False, maxOversamlingFactor=12)

    #dataset = DataAugmentationDataset(dataset, grayscale=False, augmentation_factor=8)

    doManual = False
    if doManual:
        print(f"Loading Manual Dataset")

        manualDataset = BronchosopyDataset("DatabaseManualBendOffset", transform=transform, blurSigma=0, maxEpisodes=100, shuffle=False)
        manualDataset = FilteredUpsampledDataset(manualDataset, doUpsample=True, maxOversamlingFactor=12, upsampleIndex=2)
        manualDataset = DataAugmentationDataset(manualDataset, grayscale=False, augmentation_factor=8, doStatePerturbation=False, doGoalPerturbation=False)


    datasetSize = len(dataset)
    valSize = int(0.05 * datasetSize)
    trainSize = datasetSize - valSize

    trainSubset = torch.utils.data.Subset(dataset, range(trainSize))
    valSubsetSequence = torch.utils.data.Subset(dataset, range(trainSize, datasetSize))

    trainSet, valSubsetRandom = torch.utils.data.random_split(trainSubset, [trainSize - valSize, valSize])

    batchSize = 64

    if doManual:
        trainSet = torch.utils.data.ConcatDataset([trainSet, manualDataset])

    train_loader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    val_loader_random = DataLoader(valSubsetRandom, batch_size=batchSize, shuffle=False)
    val_loader_sequence = DataLoader(valSubsetSequence, batch_size=batchSize, shuffle=False)


    sampleImage, sampleState, sampleGoal, sampleLabel, _ = dataset[0]

    numStates = len(sampleState)
    numGoal = len(sampleGoal)
    imageSize = sampleImage.shape[1]
    channels = sampleImage.shape[0]
    classes = len(sampleLabel)


    # Instantiate the model
    model = MultiInputModel(numStates=numStates, numGoal=numGoal, imageSize=imageSize, channels=channels, classes=classes)

    # Loss and optimizer
    criterion = nn.KLDivLoss(reduction='batchmean')
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loss_values = []
    random_val_loss_values = []
    seq_val_loss_values = []


    worstIndexes = defaultdict(set)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        index = 0

        all_true_labels = []
        all_predicted_labels = []



        for images, states,goals, labels, episodeAndIndex in train_loader:
            images = images.to(device)
            states = states.to(device)
            goals = goals.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            with autocast(device_type):
                outputs = model(images, states, goals)

                logOutputs = F.log_softmax(outputs, dim=1)

            #print(f"Outputs: {outputs}, Labels: {labels}")

                loss = criterion(logOutputs, labels)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            #loss.backward()
            #optimizer.step()


            running_loss += loss.item()
            index += 1

            _ , predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)  # Get the true class from one-hot encoded labels

            all_true_labels.extend(true_labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

            del images, states, goals, labels, outputs, logOutputs
            torch.cuda.empty_cache()  # Clear unused cached memory



            print(f"\rEpoch [{epoch + 1}/{epochs}], Batch [{index}/{len(train_loader)}], Loss: {running_loss / index:.4f}          ", end="")
        print("")


        train_loss = running_loss / len(train_loader)
        train_loss_values.append(train_loss)

        train_accuracy = 100 * sum(np.array(all_true_labels) == np.array(all_predicted_labels)) / len(all_true_labels)
        train_precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
        train_recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
        train_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

        #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


        #print(f"Validation random loss: {random_val_loss:.4f}, accuracy: {random_accuracy:.4f}")

        train_metrics = {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        }

        epoch_output_folder = os.path.join(output_folder, f'epoch/{epoch}')
        os.makedirs(epoch_output_folder, exist_ok=True)

        random_val_loss, random_val_metrics = validate(model, val_loader_random, criterion, device, epochOutputFolder=epoch_output_folder, epoch = epoch,prefix="random")

        # Validation for sequence samples
        seq_val_loss, seq_val_metrics = validate(model, val_loader_sequence, criterion, device, epochOutputFolder=epoch_output_folder, epoch = epoch, prefix="sequence")

        random_val_loss_values.append(random_val_loss)
        seq_val_loss_values.append(seq_val_loss)

        # Save metrics to CSV file
        save_metrics_to_csv(epoch + 1, train_metrics, random_val_metrics, seq_val_metrics, metrics_file_path)

        print(f"Training: Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1 Score: {train_metrics['f1']:.4f}")
        print(f"Random Validation: Loss: {random_val_loss:.4f}, Accuracy: {random_val_metrics['accuracy']:.4f}, "
              f"Precision: {random_val_metrics['precision']:.4f}, Recall: {random_val_metrics['recall']:.4f}, F1 Score: {random_val_metrics['f1']:.4f}")
        print(f"Sequence Validation: Loss: {seq_val_loss:.4f}, Accuracy: {seq_val_metrics['accuracy']:.4f}, "
              f"Precision: {seq_val_metrics['precision']:.4f}, Recall: {seq_val_metrics['recall']:.4f}, F1 Score: {seq_val_metrics['f1']:.4f}")

        # Save the model if the validation loss improves
        if (random_val_loss + seq_val_loss) / 2 < best_val_loss:
            best_val_loss = (random_val_loss + seq_val_loss) / 2
            torch.save(model.state_dict(), os.path.join(output_folder, modelSavePath))
            print(f"Model saved to {os.path.join(output_folder, modelSavePath)}")


        #save weights to epoch folder
        torch.save(model.state_dict(), os.path.join(epoch_output_folder, "epochModel.pth"))

        #print("")
        # Print memory summary
        #print(f"Memory Summary at the end of epoch {epoch + 1}:")
        #print(torch.cuda.memory_summary())

        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())

    # Save final loss plot
    plot_and_save_loss_curve(train_loss_values, random_val_loss_values, seq_val_loss_values, output_folder)

def validate(model, val_loader, criterion, device, device_type='cuda', epochOutputFolder="runs/implicitValidation/implicitTraining",epoch=0, prefix=""):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_true_labels = []
    all_predicted_labels = []
    all_confidences = []
    class_losses = {i: [] for i in range(6)}  # assuming 6 actions (excluding -1: No Input)

    # Create the output directory for the epoch
    

    with torch.no_grad():
        for images, states, goals, labels, _ in val_loader:
            images = images.to(device)
            states = states.to(device)
            goals = goals.to(device)
            labels = labels.to(device)

            with autocast(device_type):
                outputs = model(images, states, goals)
                logOutputs = F.log_softmax(outputs, dim=1)
                loss = criterion(logOutputs, labels)

            val_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)

            all_true_labels.extend(true_labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

            # Confidence tracking
            confidences = torch.max(F.softmax(outputs, dim=1), dim=1).values
            all_confidences.extend(confidences.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

            # Per-class loss tracking
            for i in range(6):  # Assuming you have 6 actions (from 0 to 5)
                class_mask = (true_labels == i)
                if class_mask.any():
                    class_loss = criterion(logOutputs[class_mask], labels[class_mask])
                    class_losses[i].append(class_loss.item())

            del images, states, goals, labels, outputs, logOutputs
            torch.cuda.empty_cache()  # Clear unused cached memory

    accuracy = 100 * correct / total
    precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

    # Convert NumPy arrays or values to native Python types for JSON compatibility
    class_loss_avg = {i: float(np.mean(class_losses[i])) if class_losses[i] else 0.0 for i in range(6)}
    mean_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

    # Compute confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)

    # --- Visualization 1: Confusion Matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[actionToName[i] for i in range(6)], yticklabels=[actionToName[i] for i in range(6)])
    plt.xlabel('Predicted Actions')
    plt.ylabel('True Actions')
    plt.title(f'Confusion Matrix - {prefix} - Epoch {epoch}')
    plt.savefig(os.path.join(epochOutputFolder, f'{prefix}_confusion_matrix_epoch_{epoch}.png'))
    plt.close()

    # --- Visualization 2: Class-specific Loss ---
    plt.figure(figsize=(10, 6))
    plt.bar([actionToName[i] for i in range(6)], [class_loss_avg[i] for i in range(6)], color='skyblue')
    plt.xlabel('Action Class')
    plt.ylabel('Average Loss')
    plt.title(f'Class-Specific Loss - {prefix} - Epoch {epoch}')
    plt.savefig(os.path.join(epochOutputFolder, f'{prefix}_class_specific_loss_epoch_{epoch}.png'))
    plt.close()

    # --- Visualization 3: Confidence Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_confidences, bins=20, color='orange', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'Confidence Score Distribution - {prefix} - Epoch {epoch}')
    plt.savefig(os.path.join(epochOutputFolder, f'{prefix}_confidence_distribution_epoch_{epoch}.png'))
    plt.close()

    # --- Visualization 4: Loss and Accuracy ---
    plt.figure(figsize=(8, 6))
    metrics_values = [val_loss / len(val_loader), accuracy]
    metrics_labels = ['Validation Loss', 'Accuracy (%)']
    plt.bar(metrics_labels, metrics_values, color=['red', 'green'])
    plt.title(f'Overall Validation Loss and Accuracy - {prefix} - Epoch {epoch}')
    plt.savefig(os.path.join(epochOutputFolder, f'{prefix}_loss_accuracy_epoch_{epoch}.png'))
    plt.close()

    # Save metrics and class-specific loss information
    metrics = {
        'loss': float(val_loss / len(val_loader)),  # Ensure float type
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_loss': {actionToName[i]: class_loss_avg[i] for i in range(6)},  # Use action names
        'low_confidence': mean_confidence  # Already converted to float
    }

    # Save the metrics to a JSON file with prefix and epoch
    with open(os.path.join(epochOutputFolder, f'{prefix}_metrics_epoch_{epoch}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Print metrics
    print(f"Epoch {epoch} - {prefix.capitalize()} Validation:")
    print(f"  Validation Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Class-wise Loss:")
    for action, loss in metrics['class_loss'].items():
        print(f"    {action}: {loss:.4f}")
    print(f"  Average Prediction Confidence: {metrics['low_confidence']:.4f}")

    return val_loss / len(val_loader), metrics


# Command line argument parsing
if __name__ == "__main__":
    
    main()
