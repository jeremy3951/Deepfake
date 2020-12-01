

from skimage.io import imsave
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import imageio.core.util


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
# If you want to change the default size of image saved from 160, you can
# uncomment the second line and set the parameter accordingly.
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)
# mtcnn = MTCNN(margin=40, select_largest=False, post_process=False,
# device='cuda:0', image_size=256)
  
# Directory containing images respective to each video
source_frames_folders = "D:/Celeb-DF-v2/train/yt-real"
    
# Destination location where faces cropped out from images will be saved
dest_faces_folder = "D:/Celeb-DF-v2/test"


names = []
    

c = 0
for i in os.listdir(source_frames_folders):  # video name
    
    
    for j in range(1,51): #1到100個檔名
        
        imgs = source_frames_folders +"/"+i+'/'+ str(j) + '.jpg'    
        
        if os.path.isfile(imgs)==False:
            break
            
        frame = cv2.imread(imgs)
       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = Image.fromarray(frame)
        face = mtcnn(frame)
        
        try:
            imsave(
            imgs,
            face.permute(1, 2, 0).int().numpy(),
            )
        except AttributeError:
            print("Image skipping")
    
    
        
        
        
        
        
