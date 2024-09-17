
import os
import json
import tempfile
import shutil
import numpy as np
import time
import cv2
import multiprocessing
import subprocess
from scipy.ndimage import gaussian_filter1d


from PIL import Image
from Input import Input

import compileVideo

def check_serializable(obj, path):

    if isinstance(obj, dict):
        for key, value in obj.items():
            check_serializable(value, path + f".{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            check_serializable(item, path + f"[{i}]")
    else:
        try:
            json.dumps(obj)
        except Exception as e:
            print(f"Object at path {path} is not serializable: {obj}")
            raise e

                



def saveToDiskThreadSafe(image, path):
    #signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    
    
    #convert - switch the color channels from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    
    image = Image.fromarray(image)
    image.save(path)



class Frame:
    def __init__(self, image, state, action, data, topImage=None):
        # Ensure all attributes are numpy arrays
        if not isinstance(image, np.ndarray) and not isinstance(image, str) and image is not None:
            raise ValueError(f"Image must be a numpy array or a string, its type is {type(image)}")
        if not isinstance(topImage, np.ndarray) and not isinstance(topImage, str) and topImage is not None:
            raise ValueError(f"Top image must be a numpy array or a string or None, its type is {type(topImage)}")
        

        self.image = image
        self.state = state
        self.action = action
        self.data = data
        self.topImage = topImage


    def check(self):

        check_serializable(self.data, "data")
        check_serializable(self.state, "state")
        check_serializable(self.action, "action")

    def __repr__(self):
        return f"Frame(image={self.image}, state={self.state}, action={self.action}, data={self.data}, topImage={self.topImage})"

    def copy(self):
        return Frame(
            image=self.image.copy(),
            state=self.state.copy(),
            action=self.action.copy(),
            data=self.data.copy(),
            
            topImage=self.topImage.copy()
        )

class Episode:
    



    def __init__(self, pool=None ,doAskForLabel=True, doAskForSave=True, cacheImages=True):
        

        self.doAskForLabel = doAskForLabel
        self.doAskForSave = doAskForSave
        
        self.saveEnabled = True
        #Create a new temporary directory for the episode

        self.path = tempfile.mkdtemp()
        self.dataPath = os.path.join(self.path, "00_episodeData.json")

        self.episodeProbertyLabels = {"Section": None, "Branch": None}

        self.cacheImages = cacheImages


        #Setup the episode's properties
        self.isSaved = False
        self.isLabeled = True
        self.label = None
        
        self.pool=pool


        #data
        self._images = []
        self._states = []
        self._actions = []
        self._data = []
        self._topImages = []

        self._imagesLoaded = []
        
        self.doSave=True

        self._index = 0
        self.name = None
        
        
        
        self.timeStart = time.time()
        self.timeEnd = None



    def get_frame(self, index, getImage=True):
        #print(f"Getting frame at index {index}")
        if index < 0 or index >= len(self._images):
            raise IndexError(f"Index out of range: index {index} size is {len(self._images)}")
            return None
        


        #print(f"Getting frame image: {self._images[index]}, of type {type(self._images[index])}")
        image = self._images[index]
        topImage = self._topImages[index]

        #check if either image is a string, in which case it is a path to the image, and load it
        if getImage:
            if isinstance(self._images[index], str):
                image = cv2.imread(self._images[index])

            if isinstance(self._topImages[index], str):
                topImage = cv2.imread(self._topImages[index])


            if self.cacheImages:
                self._images[index] = image
                self._topImages[index] = topImage



        frame= Frame(
            image=image,
            state=self._states[index],
            action=self._actions[index],
            data=self._data[index],
            topImage=topImage
        )

        if frame.image is None:
            print(f"Image at index {index} is None")
            return None
        return frame

    #overload the [] operator to construct and get the frame at the given index
    def __getitem__(self, index):
        return self.get_frame(index)



    def set_frame(self, index, frame, setImage=True):
        if index < 0 or index >= len(self._images):
            raise IndexError(f"Index out of range: index {index} size is {len(self._images)}")
            return None
        
        

        self._images[index] = frame.image
        self._states[index] = frame.state
        self._actions[index] = frame.action
        self._data[index] = frame.data
        self._topImages[index] = frame.topImage

        if setImage:
            imageHasChanged = not np.array_equal(self._images[index], frame.image)
            if frame.topImage is not None:
                self._topImages[index] = frame.topImage
                self.saveImage(frame.topImage, index, prefix="top")

            if imageHasChanged:
                self.saveImage(frame.image, index, prefix="broncho")  


        #overload the [] operator to set the frame at the given index
    def __setitem__(self, index, frame):

        self.set_frame(index, frame)




    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self._images):
            result = self[self._index]
            self._index += 1
            
            #print(f"Returning frame {result} at index {self._index - 1}" )
            return result, self._index - 1
        else:
            raise StopIteration

    def __len__(self):
        return len(self._images)

    def extract_model_values(self):
        """
        Extracts the 'model_value' entry from each frame's action dictionary.
        
        :return: A list of 'model_value' lists, one per frame.
        """
        model_values = []

        for index in range(len(self._images)):
            print(f"\rExtracting model values: {index+1}/{len(self._images)}", end="")
            frame = self.get_frame(index, getImage=False)
            action = frame.action  # frame[0] because __next__ returns (frame, index)
            if "model_value" in action:
                model_values.append(action["model_value"])
            else:
                model_values.append([0.0] * 6)  # Default to a zero list if not found
        print("")
        return np.array(model_values, dtype=np.float32)


    def normalize_model_values(self,model_values):
        """
        Normalizes the 'model_value' entries for each frame to ensure they sum to 1.
        
        :param model_values: A 2D numpy array where each row is a 'model_value' list for a frame.
        :return: A normalized version of model_values.
        """
        normalized_values = np.zeros_like(model_values)
        
        for i, values in enumerate(model_values):
            print(f"\rNormalizing model values: {i+1}/{len(model_values)}", end="")
            sum_values = np.sum(values)
            if sum_values > 0:
                normalized_values[i] = values / sum_values  # Normalize by dividing by sum
            else:
                normalized_values[i] = values  # If sum is zero, leave it unchanged
        print("")
        
        return normalized_values


    def gaussian_blur_model_values(self,model_values, sigma=1):
        """
        Applies a Gaussian blur to the 'model_value' lists across frames.

        :param model_values: A 2D numpy array where each row is a 'model_value' list for a frame.
        :param sigma: Standard deviation for Gaussian kernel.
        :return: A blurred version of the model_values.
        """
        # Apply Gaussian blur along the time axis (frames) for each of the 6 model_value entries
        print(f"Applying Gaussian blur with sigma={sigma} to model values")
        blurred_values = gaussian_filter1d(model_values, sigma=sigma, axis=0)
        
        #blurred_values = model_values
        
        return blurred_values


    def apply_gaussian_blur_with_normalization(self, sigma=1):
        """
        Extracts, blurs, and normalizes 'model_value' entries across frames.

        :param sigma: Standard deviation for Gaussian kernel in the Gaussian blur.
        """
        # Step 1: Extract model values from all frames
        model_values = self.extract_model_values()

        # Step 2: Normalize the extracted model values
        normalized_values = self.normalize_model_values(model_values)

        # Step 3: Apply Gaussian blur to the normalized values
        blurred_values = self.gaussian_blur_model_values(normalized_values, sigma=sigma)

        # Step 4: Normalize the blurred values again to maintain valid probabilities
        re_normalized_values = self.normalize_model_values(blurred_values)

        #for i in range(len(model_values)):
            #print(f"{i}: {re_normalized_values[i]}")
            

        # Step 5: Update the episode with the re-normalized blurred model values

        for i in range(len(self._images)):
            frame = self.get_frame(i, getImage=False)

            print(f"\rSetting blurred model values: {i+1}/{len(self._images)}", end="")
            frame.action["model_value"] = re_normalized_values[i].tolist()
            self.set_frame(i, frame, setImage=False)
        print("")


    def saveImage(self, image, index, prefix=""):
        
        imagePath = self.getImagePath(index, prefix=prefix)
        
        #print(f"Saving image of shape {image.size} to {imagePath}")
        
        
        #save the image to the disk in seperate process
        if self.pool is None:
            if isinstance(image, str):
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    
            image = Image.fromarray(image)
            image.save(imagePath)
        else:
            self.pool.apply_async(saveToDiskThreadSafe, args=(image, imagePath))
        
    def filter_frames(self, condition_fn):
        """
        Filters frames by removing those that do not meet the condition.
        
        :param condition_fn: A function that takes a frame as input and returns True if the frame should be kept, False if it should be deleted.
        """
        indices_to_remove = []  # To store indices of frames to be removed

        # Iterate through frames and check the condition function
        for index in range(len(self._images)):
            frame = self.get_frame(index)

            # If the condition function returns False, mark the frame for removal
            if not condition_fn(frame):
                indices_to_remove.append(index)
        
        # Remove the frames in reverse order to avoid index shifting issues
        for index in sorted(indices_to_remove, reverse=True):
            self._remove_frame(index)  

    def filter_frames_by_action(self):
        """
        Filters frames by removing those whose actions do not pass the hasAction() check.
        This uses the Input class to parse the action and checks if the action is valid.
        """
        indices_to_remove = []  # Store indices of frames to be removed

        # Iterate through each frame
        for index in range(len(self._images)):
            frame = self.get_frame(index)

            # Convert the frame's action (dict) into an Input object
            input_action = Input.from_dict(frame.action)

            # If input_action has no valid action, mark this frame for removal
            if not input_action.hasAction():
                indices_to_remove.append(index)

            print(f"\rFiltering frames by action: {index+1}/{len(self._images)}, to remove: {len(indices_to_remove)} to keep: {len(self._images) - len(indices_to_remove)}", end="")
        print("")
        print(f"Removing {len(indices_to_remove)} frames with invalid actions")
        # Remove frames in reverse order to avoid index shifting issues
        for index in sorted(indices_to_remove, reverse=True):
            self._remove_frame(index)  
        print(f"Filtered frames by action: {len(self._images)} frames remaining")
    
    def _remove_frame(self, index):
        """
        Helper function to remove the frame at the specified index.
        This removes associated image, state, action, and data.
        
        :param index: The index of the frame to be removed.
        """
        # Remove frame data at the specified index
        del self._images[index]
        del self._topImages[index]
        del self._states[index]
        del self._actions[index]
        del self._data[index]

        # Adjust _index for iteration if necessary
        if self._index > index:
            self._index -= 1

    def _addFrame(self, frame):


        #frame.clean()
        frame.check()


        #copy images if they are images, otherwise save the path
        if isinstance(frame.image, np.ndarray):
            self._images.append(frame.image.copy())
        else:
            self._images.append(frame.image)

        if isinstance(frame.topImage, np.ndarray):
            self._topImages.append(frame.topImage.copy())
        else:
            self._topImages.append(frame.topImage)


        self._states.append(frame.state.copy())
        self._actions.append(frame.action.copy())

        
        


        self._data.append(frame.data.copy())

        

        Index = len(self._images) - 1
        self.saveImage(frame.image,Index, prefix="broncho")
        self.saveImage(frame.topImage,Index, prefix="top")


    def append(self, frameOrImage, state=None, action=None, data=None):
        #print(f"Appending frame with image of shape {frameOrImage.shape}")
        
        frame=None
        
        if state is None: #if only one argument is given, it is a frame
            #check if the frame is a Frame object
            if not isinstance(frameOrImage, Frame):
                raise ValueError("If only one argument is given, it must be a Frame object.")
            frame=frameOrImage
        else:
            frame = Frame(image=frameOrImage, state=state, action=action, data=data)
        
        
        
        #populate the frame with meta data
        
        frame.data["index"] = len(self._images)
        
        
        currentTimestamp = time.time()
        
        
        frame.data["Timestamp"] = f"{(currentTimestamp - self.timeStart):.3f}"
        
        
        frame.data["Raw_Timestamp"] = currentTimestamp
        frame.data["timeReadable"] = time.strftime("%Y-%m-%d %H:%M:%S")+f".{str(currentTimestamp%1)[2:5]}"
        
        lastTimestamp = self._data[-1]["Raw_Timestamp"] if len(self._data) > 0 else None
        frame.data["timeDelta"] = currentTimestamp - lastTimestamp if lastTimestamp is not None else 0
        
        
        
           
        self._addFrame(frame)
            

    def getImagePath(self, index,prefix): #returns the path of the image at the given index, usign zfill to pad the index with zeros
        return os.path.join(self.path, self.getName(index,prefix))


    def getName(self,index,prefix):
        return f"{prefix}{str(index).zfill(5)}.png"

    def __del__(self):
        try:
            shutil.rmtree(self.path)
            print(f"Temporary folder at {self.path} deleted.")
        except Exception as e:
            #print(f"Error deleting temporary folder: {e}")
            pass


    def askForLabel(self):
        
        
        if self.doAskForSave:
            while True:
                saveInput="y"
                if not self.doSave:
                    saveInput = input("Do you want to save the episode? (y/n): ")
                if saveInput.lower() == "n":
                    return
                elif saveInput.lower() == "y":
                    self.doSave = True
                    
                    if not self.isLabeled:
                        for key in self.episodeProbertyLabels.keys():
                            value = input(f"Enter the {key}: ")
                            self.episodeProbertyLabels[key] = value
                            
                        self.isLabeled = True
                        
                    
                    return
                    
                else:
                    print("Invalid input. Please enter 'y' or 'n'")
            
                
            

    def analyseDict(self, dictionary, indent=0):

        for key, value in dictionary.items():
            
            if isinstance(value, dict):
                print(f"{' '*indent} dict {key} :")
                self.analyseDict(value, indent=indent+4)
            elif isinstance(value, list):
                print(f"{' '*indent} list {key} :")
                for item in value:
                    if isinstance(item, dict):
                        self.analyseDict(item, indent=indent+4)
                    else:
                        print(f"{' '*(indent+4)}{type(item)} : {item}")
            else:
                print(f"{' '*(indent+4)}{type(value)} {key} : {value}")




    def save(self, directory=None):
        
        #ask for the label
        doSave = self.askForLabel()
        if self.doSave is False:
            return
        

        #construct the dictionary to save

        episodeData = {}

        #construct header data
        header = {}
        
        
        self.timeEnd = self._data[-1]["Raw_Timestamp"]
        
        header["timeStart"] = self.timeStart
        header["timeEnd"] = self.timeEnd
        header["duration"] = self.timeEnd - self.timeStart
        header["timeStartReadable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timeStart))+f".{str(self.timeStart%1)[2:5]}"
        header["timeEndReadable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timeEnd))+f".{str(self.timeEnd%1)[2:5]}"
        
        
        
        for key, value in self.episodeProbertyLabels.items():
            header[key] = value
        
        episodeData["header"] = header

        #save the frames
        frames = {}
        
        print(f"Compiling episode with {len(self._images)} frames")

        for frame, index in self:
            

            print(f"\rSaving frame {index} of {len(self._images)}", end="")
            
            #print(f"Saving frame {frame} at index {index}")
            frameData = {}
            frameData["state"] = frame.state
            frameData["action"] = frame.action
            frameData["data"] = frame.data
            frameData["imagePath"] = self.getName(index, prefix="broncho")
            frameData["topImagePath"] = self.getName(index, prefix="top")
            
            frames[index] = frameData
            
            
        episodeData["frames"] = frames
        
        
        
        #print(f"Frame 0 example: {frames[0]}")
        #save the episode data to a json file
        

        #find_non_serializable(self, episodeData)
        

        print("\nSaving episode data to json file")

        with open(self.dataPath, 'w') as f:
            json.dump(episodeData, f, indent=4)
            
            
        #save the episode to a zip file
        
        name = time.strftime("20%y-%m-%d - %H-%M-%S", time.localtime(self.timeStart))
        self.name = name
        
        #if a directory is given, save the episode there
        
        
        #print the objects in the directory
        
        
        
        print("Compresing episode to zip file")
        name = os.path.join(directory, name)
        if directory is not None:
            
            
            print(f"Saving episode to {name}")
            shutil.make_archive(name, 'zip', self.path)





        #render the video
        #print("Rendering video")



        #compileVideo.create_videos_from_folder(name, create_combined=True)
        #name = name + ".zip"
        #render the video in a new process using pool
        #self.pool.apply_async(compileVideo.create_videos_from_folder, args=(name, True))
    

    #create a Episode.load method that loads an episode from a given path
    @classmethod
    def load(cls, path, pool=None, cacheImages=True):
        
        #create a new episode object
        episode = cls(pool, cacheImages=cacheImages)

        #find name from path: file name without extension
        episode.name = os.path.splitext(os.path.basename(path))[0]

        #extract the zip file to the temporary folder created by the episode
        shutil.unpack_archive(path, episode.path)

        #load the episode data from the json file

        with open(episode.dataPath, 'r') as f:
            episodeData = json.load(f)



        #load the header data
        header = episodeData["header"]
        episode.timeStart = header["timeStart"]
        episode.timeEnd = header["timeEnd"]
        #episode.duration = header["duration"]
        
        episode.episodeProbertyLabels = {key: header[key] for key in episode.episodeProbertyLabels.keys()}



        #load the frames

        frames = episodeData["frames"]

        for index, frameData in frames.items():
            #load the frame data
            state = frameData["state"]
            action = frameData["action"]
            #print(action)

            data = frameData["data"]
            imagePath = os.path.join(episode.path, frameData["imagePath"])
            topImagePath = os.path.join(episode.path, frameData["topImagePath"])
            #load the image
            #image = cv2.imread(imagePath)
            #topImage = cv2.imread(topImagePath)
            image = imagePath
            topImage = topImagePath

            #make sure both images exists
            if not os.path.exists(imagePath) or not os.path.exists(topImagePath):
                print(f"Image at {imagePath} or top image at {topImagePath} does not exist")
                break


            #print(f"Loading frame {index} with image of shape {image.shape}")
            #create a new frame
            frame = Frame(image=image, state=state, action=action, data=data, topImage=topImage)
            episode._addFrame(frame)


        return episode


        


    

    def clear(self):

        




        #clear the episode's data and empty the temporary folder



        self._images = []
        self._states = []
        self._actions = []
        self._data = []


        try:
            #delete the temporary folder, if it exists  
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
                #print(f"Temporary folder at {self.path} deleted.")
            else: 
                print(f"Temporary folder at {self.path} already deleted")
        except Exception as e:
            print(f"Error deleting temporary folder: {e}")




class EpisodeManager:

    def __init__(self, mode = "Recording", saveLocation="Database/", loadLocation="Database/", multiProcessing=False):
        self.currentEpisode = None


        #set mode to lower case
        mode = mode.lower()

        self.mode = mode

        self.defaultSaveLocation = saveLocation
        self.defaultLoadLocation = loadLocation
        
        self.multiProcessing = multiProcessing
        if self.multiProcessing:
            self.pool = multiprocessing.Pool(processes=8)
        else:
            self.pool = None

        self.currentIndex = -1

        


        self.findAllEpisodes()


    def findAllEpisodes(self, episodeDatabasePath=None): #returns a list of all episodes in the database. Each episode is a zip file
        if episodeDatabasePath is None:
            episodeDatabasePath = self.defaultLoadLocation


        episodes = []
        #Search each folder and file in the database
        for root, dirs, files in os.walk(episodeDatabasePath):
            for file in files:
                if file.endswith(".zip"):
                    episodes.append(os.path.join(root, file))

        self.allEpisodes = episodes
        return episodes
    

    def loadEpisode(self, indexOrPath, cacheImages=True):
        
        if isinstance(indexOrPath, int):
            #check if the index is in range
            if indexOrPath < 0 or indexOrPath >= len(self.allEpisodes):
                return None
            path = self.allEpisodes[indexOrPath]
        else:
            path = indexOrPath

        self.currentEpisode = Episode.load(path, self.pool, cacheImages=cacheImages)
        return self.currentEpisode
    
    def doShuffleEpisodes(self):
        #shuffle the list of strings that represent the episodes
        
        #get new index order
        print("Shuffling episodes")
        newOrder = np.random.permutation(len(self.allEpisodes))

        #reorder the list of episodes
        self.allEpisodes = [self.allEpisodes[i] for i in newOrder]



    def loadAllEpisodes(self, maxEpisodes=None, cacheImages=True, shuffle=False):
        episodes = []
        episodeFrameIndexStart = []

        if shuffle:
            self.doShuffleEpisodes()

        totalFrames = 0
        for i in range(len(self.allEpisodes)):
            if maxEpisodes is not None and i >= maxEpisodes:
                break
            
            print(f"\rLoading episode {i} of {len(self.allEpisodes)} with name: {self.allEpisodes[i]}", end="")

            episode = self.loadEpisode(i, cacheImages=cacheImages)
            episodes.append(episode)
            episodeFrameIndexStart.append(totalFrames)
            totalFrames += len(episode)

        print("")
        print(f"Loaded {len(episodes)} episodes with a total of {totalFrames} frames")
        print("")
        
        
        return episodes, totalFrames, episodeFrameIndexStart


    def newEpisode(self, force=False):
        if self.currentEpisode is not None:
            if force:
                self.endEpisode(discard=True)
            else:
                return
        
        self.currentEpisode = Episode(self.pool)
        return
        

        
                
    def endEpisode(self, discard=False, saveLocation=None):
        if self.currentEpisode is None:
            return
        
        if saveLocation is None:
            saveLocation = self.defaultSaveLocation
        
        #save the episode to the database
        if discard is True:
            self.currentEpisode.clear()
            self.currentEpisode = None
            return
        
        

        if self.currentEpisode.isSaved is False:
            
            self.currentEpisode.save(saveLocation)

        self.currentEpisode.clear()
        self.currentEpisode = None

        self.findAllEpisodes()
        
        
    def getCurrentEpisode(self):
        return self.currentEpisode

    def nextEpisode(self, index=None):
        if index is not None:
            self.currentIndex = index-1


        if self.mode == "recording":    
            self.endEpisode()
            self.newEpisode()
        elif self.mode == "labelling": 
            self.endEpisode() 
            self.currentIndex += 1
            if self.currentIndex >= len(self.allEpisodes):
                self.currentIndex = len(self.allEpisodes) - 1

            self.loadEpisode(self.currentIndex)
        elif self.mode == "read":
            self.endEpisode(discard=True)
            self.currentIndex += 1
            if self.currentIndex >= len(self.allEpisodes):
                self.currentIndex = len(self.allEpisodes) - 1

            self.loadEpisode(self.currentIndex)
        else:
            print("Invalid mode")
            
            return None

        return self.currentEpisode


    def previousEpisode(self):
        self.endEpisode()
        self.currentIndex -= 1
        if self.currentIndex < 0:
            self.currentIndex = 0

        self.loadEpisode(self.currentIndex)

    def hasEpisode(self):
        return self.currentEpisode is not None
    
    def hasNextEpisode(self):
        return self.currentIndex < len(self.allEpisodes) - 1
    
    
    #pass append on to current episode
    def append(self, *args, **kwargs):
        #print(f"Appending to episode with args {args} and kwargs {kwargs}")
        if self.currentEpisode is not None:
            self.currentEpisode.append(*args, **kwargs)
        else:
            print("No current episode")

    
        

            
            




        

