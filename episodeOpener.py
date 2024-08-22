







from DataHandling.Episode import EpisodeManager, Episode









def main():
    episodeManager = EpisodeManager(mode = "Labelling", loadLocation="DatabaseLabelled/", saveLocation="DatabaseLabelled/")

    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()
        print(f"Episode: {episodeManager.currentIndex}")


if __name__ == "__main__":
    main()