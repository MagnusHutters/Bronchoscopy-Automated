


import numpy as np
from branchModelTracker import BranchModelTracker
from DataHandling.Episode import EpisodeManager, Episode

def main():


    episodeManager = EpisodeManager(mode = "labelling", loadLocation= "DatabaseLabelled", saveLocation= "DatabaseLabelledPost")

    print(f"Number of episodes: {episodeManager.numEpisodes}")


    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()

        print(f"Epsiode {episodeManager.currentIndex}")



        episode.filter_frames_by_action()


    episodeManager.endEpisode()




if __name__ == "__main__":
    main()