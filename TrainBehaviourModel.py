import argparse
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


from DataHandling.Episode import Episode, EpisodeManager


# Custom Dataset class
class BronchosopyDataset(Dataset):
    def __init__(self, databasePath, transform=None):
        self.episodeManager = EpisodeManager(mode = "read", loadLocation = databasePath)


        self.episodes, self.lenght, self.episodeFrameIndexStart = self.episodeManager.loadAllEpisodes(cacheImages=True, maxEpisodes=100)
        self.transform = transform

        self.actionToIndex = {"u": 0, "d": 1, "l": 2, "r": 3, "f": 4, "b": 5}
        

    def __len__(self):
        return self.lenght


    def getEpisodeAndSubIndex(self, index):
        episodeIndex = 0
        subIndex = 0

        for i in range(len(self.episodeFrameIndexStart)-1, 0, -1):
            if index >= self.episodeFrameIndexStart[i]:
                episodeIndex = i
                subIndex = index - self.episodeFrameIndexStart[i]
                break

        return episodeIndex, subIndex


    def __getitem__(self, idx):
        
        episodeIndex, subIndex = self.getEpisodeAndSubIndex(idx)

        #print(f"Getting index {idx} episodeIndex {episodeIndex} subIndex {subIndex} using frameStart {self.episodeFrameIndexStart}")

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
        goalInput = torch.tensor(goalCenter + goalSize, dtype=torch.float)

        action = frame.action["char_value"] # char value u, d, l, r, f, b or none
        oneHotAction = torch.zeros(6, dtype=torch.float)

        if action in self.actionToIndex:
            oneHotAction[self.actionToIndex[action]] = 1
        
            
        


        
        return image, stateInput, goalInput, oneHotAction


class FilteredUpsampledDataset(Dataset):
    def __init__(self, originalDataset, maxOversamlingFactor=12):

        print("Filtering and upsampling dataset")
        self.primaryDataset = originalDataset

        self.maxOversamlingFactor = maxOversamlingFactor

        validIndices = self.filterInvalidIndices()

        classCounts = self.countClasses(validIndices)

        oversamplingFactors = self.calculateOversamplingFactors(classCounts)

        oversampledIndices = self.oversampleIndices(validIndices, oversamplingFactors)

        self.oversampledIndices = oversampledIndices

    def filterInvalidIndices(self):
        validIndices = []

        for i in range(len(self.primaryDataset)):
            image, state, goal, label = self.primaryDataset[i]

            #label is one-hot encoded, invalid if all zeros
            if torch.sum(label) > 0:
                validIndices.append(i)

            print(f"\rFiltering invalid indices {i}/{len(self.primaryDataset)} - valid: {len(validIndices)}, invalid: {i - len(validIndices)}    ", end="")
        print("")


        return validIndices
    
    def countClasses(self, validIndices):
        classCounts = {}

        for i, idx in enumerate(validIndices):
            image, state, goal, label = self.primaryDataset[idx]

            classId = torch.argmax(label).item()

            if classId not in classCounts:
                classCounts[classId] = 0
            classCounts[classId] += 1

            print(f"\rCounting classes {i}/{len(validIndices)}", end="")

        print("")
        for classId in classCounts:
            print(f"Class {classId} count: {classCounts[classId]}")
        return classCounts
    
    def calculateOversamplingFactors(self, classCounts):
        oversamplingFactors = {}

        #find factor such that each class has equal amount of samples

        #maxCount = max(classCounts.values())

        #second largest count
        maxCount = sorted(classCounts.values())[-1]

        for classId in classCounts:
            count = classCounts[classId]
            factor = maxCount / count

            if factor > self.maxOversamlingFactor:
                factor = self.maxOversamlingFactor

            oversamplingFactors[classId] = factor
        
        print("Sampling factors")
        for classId in oversamplingFactors:
            print(f"Class {classId} factor: {oversamplingFactors[classId]}")

        return oversamplingFactors
    
    def oversampleIndices(self, validIndices, oversamplingFactors):
        oversampledIndices = []

        accumulator = {classId: 0 for classId in oversamplingFactors}
        counts = {classId: 0 for classId in oversamplingFactors}

        for idx in validIndices:
            image, state, goal, label = self.primaryDataset[idx]

            classId = torch.argmax(label).item()

            factor = oversamplingFactors[classId]

            toAdd = factor+accumulator[classId]

            toAddActual = int(toAdd)
            remainder = toAdd - toAddActual
            accumulator[classId] = remainder

            oversampledIndices.extend([idx] * toAddActual)
            counts[classId] += toAddActual

            print(f"\rOversampling indices {idx}/{len(validIndices)}, Oversampled {len(oversampledIndices)}          ", end="")
        print("")
        print("Counts After Resampling")
        for classId in counts:
            print(f"Class {classId} count: {counts[classId]}")

        return oversampledIndices
    
    def __len__(self):
        return len(self.oversampledIndices)
    
    def __getitem__(self, idx):
        image, state, goal, label = self.primaryDataset[self.oversampledIndices[idx]]
        #label to index
        classId = torch.argmax(label).item()
        return image, state, goal, classId



# Multi-input CNN model
class MultiInputModel(nn.Module):
    def __init__(self, numStates=3, numGoal=4, imageSize=50, channels=1, classes=6):
        super(MultiInputModel, self).__init__()
        
        # Convolutional layers for the image input
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
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

        flattened_size = 64 * pool3_size * pool3_size

        self.fc1_state = nn.Linear(in_features=numStates, out_features=32)
        self.fc1_goal = nn.Linear(in_features=numGoal, out_features=32)

        concatenated_size = 32 + 32 + flattened_size
        
        # Fully connected layers for the combined output
        self.fc1_combined = nn.Linear(in_features=concatenated_size, out_features=512)
        self.fc2_combined = nn.Linear(in_features=512, out_features=256)
        self.fc3_combined = nn.Linear(in_features=256, out_features=128)
        self.fc4_combined = nn.Linear(in_features=128, out_features=classes)



        

    def forward(self, image, state, goal):
        # CNN pathway for image input
        x1 = self.pool(F.relu(self.conv1(image)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, x1.size(1) * x1.size(2) * x1.size(3))

        state = F.relu(self.fc1_state(state))
        goal = F.relu(self.fc1_goal(goal))

        x = torch.cat((x1, goal, state), dim=1)

        x = F.relu(self.fc1_combined(x))
        x = F.relu(self.fc2_combined(x))
        x = F.relu(self.fc3_combined(x))
        x = self.fc4_combined(x)


        
        return x

# Main function to handle training
def main(epochs = 50, learningRate = 0.0001, modelSavePath = "model.pth"):
    # Image transformation
    transform = transforms.Compose([
        #transforms.Grayscale(),                     # Convert images to grayscale
        transforms.Resize((50, 50)),                # Resize images to 50x50 pixels
        transforms.ToTensor(),                      # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))         # Normalize images
    ])

    # Load your dataset
    # Assuming image_paths, extra_features, and labels are preloaded lists
    # Example: Replace these with your actual data loading logic

    # Create the custom dataset and DataLoader
    dataset = BronchosopyDataset("DatabaseLabelled", transform=transform)

    dataset = FilteredUpsampledDataset(dataset)

    datasetSize = len(dataset)
    valSize = int(0.1 * datasetSize)
    trainSize = datasetSize - valSize

    trainSubset = torch.utils.data.Subset(dataset, range(trainSize))
    valSubsetSequence = torch.utils.data.Subset(dataset, range(trainSize, datasetSize))

    trainSubset, valSubsetRandom = torch.utils.data.random_split(trainSubset, [trainSize - valSize, valSize])

    train_loader = DataLoader(trainSubset, batch_size=64, shuffle=True)
    val_loader_random = DataLoader(valSubsetRandom, batch_size=64, shuffle=False)
    val_loader_sequence = DataLoader(valSubsetSequence, batch_size=64, shuffle=False)



    # Instantiate the model
    model = MultiInputModel(numStates=3, numGoal=4, imageSize=50, channels=3, classes=6)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        index = 0
        for images, states,goals, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, states, goals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            index += 1
            print(f"\rEpoch [{epoch + 1}/{epochs}], Batch [{index}/{len(train_loader)}], Loss: {running_loss / index:.4f}                ", end="")
        print("")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation loop
        seq_val_loss, seq_accuracy = validate(model, val_loader_sequence, criterion)

        print(f"Validation sequence loss: {seq_val_loss:.4f}, accuracy: {seq_accuracy:.4f}")

        random_val_loss, random_accuracy = validate(model, val_loader_random, criterion)

        print(f"Validation random loss: {random_val_loss:.4f}, accuracy: {random_accuracy:.4f}")



        total_val_loss = (seq_val_loss + random_val_loss) / 2
        # Save the model if the validation loss has decreased
        if epoch == 0 or total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            best_accuracy = seq_accuracy
            torch.save(model.state_dict(), modelSavePath)
            print(f"Model saved to {modelSavePath}")


    # Save the trained model
    torch.save(model.state_dict(), modelSavePath)
    print(f"Model saved to {modelSavePath}")


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, states, goals, labels in val_loader:
            outputs = model(images, states, goals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


# Command line argument parsing
if __name__ == "__main__":
    
    main()
