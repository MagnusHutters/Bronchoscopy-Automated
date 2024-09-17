

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from branchModelTracker import BranchModelTracker
from DataHandling.Episode import EpisodeManager, Episode

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transforms for EfficientNet and MobileNet
preprocess = transforms.Compose([
    transforms.Resize(224),  # Resize image to 224x224 for both models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

# Load EfficientNet and MobileNet
efficientnet = models.efficientnet_b0(pretrained=True).to(device)
mobilenet = models.mobilenet_v2(pretrained=True).to(device)

# Remove final classification layers to get features
efficientnet.classifier = nn.Identity()
mobilenet.classifier = nn.Identity()

# Set models to evaluation mode (no gradient calculation)
efficientnet.eval()
mobilenet.eval()

def extract_features(image, model):
    """Helper function to extract features from a model."""
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        features = model(image_tensor).squeeze(0)  # Remove batch dimension
    return features.cpu().numpy()  # Move back to CPU for easy handling in numpy

def main():
    episodeManager = EpisodeManager(mode="labelling", loadLocation="DatabaseLabelled", saveLocation="DatabaseLabelledWithFeatures")

    print(f"Number of episodes: {len(episodeManager.allEpisodes)}")

    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()

        print(f"Episode {episodeManager.currentIndex}/{len(episodeManager.allEpisodes)}")

        for frame, index in episode:

            print(f"\rProcessing frame {index}/{len(episode)}", end="")
            image = frame.image  # Extract the 400x400 image from the frame
            
            # Convert to PIL Image if it's not already
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # Extract features from EfficientNet and MobileNet
            EfficientNetFeatures = extract_features(image, efficientnet).tolist()
            MobileNetFeatures = extract_features(image, mobilenet).tolist()

            # Convert features to comma-separated strings
            EfficientNetFeaturesString = ','.join(map(str, EfficientNetFeatures))
            MobileNetFeaturesString = ','.join(map(str, MobileNetFeatures))

            # Store features in the frame data
            frame.data["EfficientNetFeatures"] = EfficientNetFeaturesString
            frame.data["MobileNetFeatures"] = MobileNetFeaturesString
        print("")


    episodeManager.endEpisode()

if __name__ == "__main__":
    main()
