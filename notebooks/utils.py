#imports
import os
import numpy as np
import cv2
from skimage import io 
from matplotlib import pyplot as plt

#paths
SCENE = "pedestrians"
SRC = f"../data/{SCENE}/input"
FRAMES_dir = f"../data/{SCENE}/frames"
MASKS_dir = f"../data/{SCENE}/masks"
OUTPUTS_dir = f"../outputs/{SCENE}"
CLICK_file = f"../data/{SCENE}/click.json"

#imshow from TDS
def imshow(
    images: list[np.ndarray] | np.ndarray,
    titles: list[str] | str | None = None,
    nrows: int = 0,
    ncols: int = 0,
    figsize: (float, float) = (10, 4),
):
    """Plot a multiple images with titles.

    Parameters
    ----------
    images : image list
    titles : title list
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    figsize : size of the figure
    """
    if not isinstance(images, list):
        images = [images]
    if titles is not None and not isinstance(titles, list):
        titles = [titles]

    if ncols == 0 and nrows == 0:
        ncols = len(images)
        nrows = 1
    if ncols == 0:
        ncols = len(images) // nrows
    if nrows == 0:
        nrows = len(images) // ncols

    fig, axeslist = plt.subplots(
        ncols=ncols, nrows=nrows, squeeze=False, figsize=figsize
    )

    for i, image in enumerate(images):
        axeslist.ravel()[i].imshow(image, cmap=plt.gray(), vmin=0, vmax=255)
        if titles is not None:
            axeslist.ravel()[i].set_title(titles[i])
        axeslist.ravel()[i].set_axis_off()
    plt.tight_layout()  # optional

    return fig, axeslist

#Read frames 
#check if it is a video :
def is_video(source):
    return source.endswith((".mp4", ".mov", ".avi"))

#open a video : cv2.VideoCapture(source)
#close a video : video.release()
#read a frame : video.read()
#go at k-th frame : video.set(cv2.CAP_PROP_POS_FRAMES, k)  

#count number of frames
def count_frames(source):
    if is_video(source):
        video = cv2.VideoCapture(source)
        n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return n
    return len(os.listdir(source)) #count files if not a video

#read the k-th frame, for a video: opens file jumps to frame k and reads it and converts to RGB, 
#for a folder: reads the k-th file in the folder
def read_frame(source, k):
    if is_video(source):
        video = cv2.VideoCapture(source)
        video.set(cv2.CAP_PROP_POS_FRAMES, k)
        _, frame = video.read()
        video.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #else : take the list of frames
    names = sorted(os.listdir(source))
    return io.imread(f"{source}/{names[k]}")


# Takes RGB frame and a binary mask, returns a copy where the masked pixels are blended 50/50 with red. 
def red_overlay(frame_rgb, mask, alpha=0.5):
    """Blend the mask pixels with red (frame in RGB)."""
    out = frame_rgb.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array([255, 0, 0])).astype(np.uint8)
    return out

# This function loads a frame from the FRAMES_dir in RGB format.
def load_frame(i):
    return io.imread(f"{FRAMES_dir}/{i}.jpg")

#load mask 
def load_mask(i) :
    #read mask i saved by SAM2 
    return np.load(f"{MASKS_dir}/{i}.npy")

#save video 
def save_video(frames, path, fps=30):
    #save list of rgb frames as mp4 video 
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   # opencv writes BGR
    writer.release()