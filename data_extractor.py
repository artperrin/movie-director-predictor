import os
import cv2
import config
import logging

logging.getLogger().setLevel(logging.INFO)

class movie:
    """Class to handle a movie
    """
    parent_dir = config.DATA_DIR
    src_dir = config.SRC_DIR

    def __init__(self, movie_name='NR', director='NR'):
        self.name = movie_name
        self.dir = director

    def get_total_frames(self):
        """method to obtain the total number of frames of the video

        Returns
        -------
        int
            number of frames of the video
        """
        filepath = os.path.join(self.src_dir, self.name)
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
        filepath = os.path.join(self.src_dir, self.name)
        cap = cv2.VideoCapture(filepath)
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
            if self.img == None:
                logging.warning("No screenshot to save...")

            folder = os.path.join(self.parent_dir, self.dir)
            path = os.path.join(folder, self.name[:-4])
            try:
                os.mkdir(folder)
                cv2.imwrite(path+'.jpg', self.img)
                logging.debug(f'New screenshot saved in the created directory : {folder}.')
            except FileExistsError:
                cv2.imwrite(path+'.jpg', self.img)
                logging.debug(f'New screenshot saved in the already existing directory : {folder}.')