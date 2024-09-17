


import numpy as np
from branchModelTracker import BranchModelTracker
from DataHandling.Episode import EpisodeManager, Episode

def main():


    episodeManager = EpisodeManager(mode = "labelling", loadLocation= "DatabaseLabelled", saveLocation= "DatabaseLabelledWithFeatures")

    print(f"Number of episodes: {len(episodeManager.allEpisodes)}")


    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()

        print(f"Epsiode {episodeManager.currentIndex}/{len(episodeManager.allEpisodes)}")

        for frame, index in episode:

            image = frame.image #extract the 400x400 image from the frame




            EfficientNetFeatures = [] #extracted features from EfficientNet - a list of floats
            MobileNetFeatures = [] #extracted features from MobileNet - a list of floats


            EfficientNetFeaturesString = ','.join(map(str, EfficientNetFeatures))
            MobileNetFeaturesString = ','.join(map(str, MobileNetFeatures))

            frame.data["EfficientNetFeatures"] = EfficientNetFeaturesString
            frame.data["MobileNetFeatures"] = MobileNetFeaturesString

            



        episode.filter_frames_by_action()


    episodeManager.endEpisode()




if __name__ == "__main__":
    main()