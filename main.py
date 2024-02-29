
import time

from DataHandling.Episode import Episode
from PIL import Image, ImageDraw







episode1 = Episode()
episode2 = Episode()
episode3 = Episode()
episode4 = Episode()
episode5 = Episode()





for i in range(10):

    #Create blank PIL image with white background
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"Frame {i}", fill="black")



    episode1.addFrame(image)
    episode2.addFrame(image)
    episode3.addFrame(image)
    episode4.addFrame(image)
    episode5.addFrame(image)



episode1.saveEpisode()
episode2.saveEpisode()
episode3.saveEpisode()
episode4.saveEpisode()
episode5.saveEpisode()



while(True):
    time.sleep(0.1)







