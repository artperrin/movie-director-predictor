import os
import cv2
import config
import logging
import random
import time

logging.getLogger().setLevel(logging.INFO)

class Movie:

    """Class to handle a movie
    """
    parent_dir = config.TRN_DIR

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def define(self, movie_name='NR', director='NR'):
        self.name = movie_name
        self.dir = director

    def get_total_frames(self):
        """method to obtain the total number of frames of the video

        Returns
        -------
        int
            number of frames of the video
        """
        filepath = os.path.normpath(self.src_dir)
        cap = cv2.VideoCapture(filepath)
        return cap.get(7)

    def take_screenshot(self, nframe):
        """stores a screenshot from the video at the given frame

        Parameters
        ----------
        nframe : int
            frame number to freeze

        Returns
        -------
        np.array
            screenshot as an image (BGR)
        """
        filepath = os.path.normpath(self.src_dir)
        cap = cv2.VideoCapture(filepath)
        self.frame = nframe 
        cap.set(1, nframe)
        _, self.img = cap.read()
        self.img = cv2.resize(self.img, config.IMG_SIZE)
        logging.debug("Screenshot of frame number {nframe} out of {total_frames} taken.")
        return self.img

    def save_screenshot(self):
        """saves the screenshot in the right directory for the training program to work
        """
        if self.dir=='NR':
            logging.error("Cannot save the screenshot : no director.")
        else:
            if self.img.size == 0:
                logging.warning("No screenshot to save...")

            folder = os.path.join(self.parent_dir, self.dir)
            path = os.path.join(folder, self.name[:-4])
            try:
                os.mkdir(folder)
                cv2.imwrite(path+str(self.frame)+'.jpg', self.img)
                logging.debug(f'New screenshot saved in the created directory : {folder}.')
            except FileExistsError:
                cv2.imwrite(path+str(self.frame)+'.jpg', self.img)
                logging.debug(f'New screenshot saved in the already existing directory : {folder}.')

if __name__ == "__main__":
    start = time.time()
    # my library is luckily organized by director
    src_dir = 'E:\\Films\\Edgar Wright\\'
    # director's name to create folder
    real = 'wright'

    src_dir = os.path.normpath(src_dir)

    # enumerate the files found on the src_dir directory
    filenames = []
    for root, directories, files in os.walk(src_dir, topdown=False):
        for name in files:
            filenames.append(os.path.join(root, name))
    logging.info(f"Found {len(filenames)} files...")

    # write in the terminal how many screenshots per movie you want
    ScreenShotNB = input('How many screenshots per movie ? ')
    ScreenShotNB = int(ScreenShotNB)
    count = 0;
    # for each film, create a instance of Movie
    for i in range(len(filenames)):
        vid = filenames[i]
        mov = Movie(vid)
        name = 'movie_'+str(i)+'.mp4'
        mov.define(name, real)
        total = mov.get_total_frames()
        # generate a random number (the frame to be frozen) between 15% and 85% of the movie (to avoid credentials)
        for k in range(ScreenShotNB):
            rd = random.randint(int(15/100*total), int(85/100*total))
            mov.take_screenshot(rd)
            mov.save_screenshot()
            count += 1
            print(f"Generated {count} screenshots out of {len(filenames)*ScreenShotNB}...", end="\r", flush=True)
    print('', end='\n')
    
    logging.info(f"Successfuly wrote {count} screenshots within {round(time.time()-start)} seconds !")