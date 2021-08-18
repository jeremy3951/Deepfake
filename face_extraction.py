from skimage.io import imsave
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
from pathlib import Path


def show_img(img):
    cv2.imshow(winname="Fa", mat= img)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()


mtcnn = MTCNN(
    margin=50,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)

# =============================================================================
# source_frames_folders = Path(r'C:\Users\jeremy\Desktop\2021DF\model\666\os\000_003')
#     
# 
# video = [x for x in source_frames_folders.iterdir()]
# 
# problem = []  # some videos which is failed
# 
# dst = r'C:\Users\jeremy\Desktop\2021DF\model\666\os\000_003\tt\546.jpg'
# =============================================================================

    
# =============================================================================
# for i in video: 
#     
#     frames = [str(x) for x in i.iterdir()]   
#     
#     for imgs in frames: 
#         
#         
#         print(imgs)
#         frame = Image.open(imgs).convert('RGB')
#         
#         try:
#             face = mtcnn(frame)
#             
#         except:
#             problem.append(imgs)
#             pass
#                 
#         try:
#             imsave(
#             dst
#             , face.permute(1, 2, 0).int().numpy(),
#             )
# 
#         except AttributeError:
#             problem.append(imgs)
#             print("Image skipping")
# =============================================================================
        

dst = r'C:\Users\jeremy\Desktop\2021DF\model\666\os\000_003\tt\546.jpg'

imgs = r'C:\Users\jeremy\Desktop\2021DF\model\666\os\000_003\1.jpg'
frame = Image.open(imgs).convert('RGB')

try:
    face = mtcnn(frame)
    
except:
    problem.append(imgs)
    pass
        
try:
    imsave(
    dst
    , face.permute(1, 2, 0).int().numpy(),
    )

except AttributeError:
    problem.append(imgs)
    print("Image skipping")
