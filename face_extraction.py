  
from skimage.io import imsave
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os

def show_img(img):
    cv2.imshow(winname="Fa", mat= img)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()


mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)

source_frames_folders = r'D:\FF++\data\test\FaceShifter'
    

names = []
problem = []  # some videos which is failed


for i in os.listdir(source_frames_folders):  # video name
    names.append(i)
    
for i in range(1 , 140): 
    path = os.path.join(source_frames_folders , names[i])
    
    for j in os.listdir(path): #1到100個檔名
    
        imgs = os.path.join(path , j)
        
        if os.path.isfile(imgs)==False:
            break
            
        frame = cv2.imread(imgs)
       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = Image.fromarray(frame)
        try:
            face = mtcnn(frame)
            
        except:
            problem.append(names[i])
            pass
        
        try:
            imsave(
            imgs
            , face.permute(1, 2, 0).int().numpy(),
            )

        except AttributeError:
            print("Image skipping")


