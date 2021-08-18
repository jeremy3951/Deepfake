import cv2 
import dlib
import numpy as np
import os

def show_img(img):
    cv2.imshow(winname="Fa", mat= img)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()


def find_landmark( path):
    
    file_path = 'C:/Users/jeremy/anaconda3/envs/dlib/Lib/site-packages/dlib/examples/shape_predictor_68_face_landmarks.dat'
    img = cv2.imread(path)
    img = cv2.resize(img , (250,250))     #resize or not
    
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(file_path)
    
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray )
    
    for face in faces:    
        x1 = face.left() 
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        

    landmarks = predictor(image=gray, box= face) 
    xy_list = []

    for n in range(0,68):  
        x = landmarks.part(n).x   
        y = landmarks.part(n).y    
        xy_list.append((x,y))
    
    xy_array = np.array(xy_list)
    return xy_array , xy_list , img

def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)


    sense1 = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)
    sense2 = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)
    
    cv2.fillConvexPoly(sense1, cv2.convexHull(
        np.concatenate((int_lmrks[36:40],
                        #int_lmrks[17:22],
                        #int_lmrks[22:27],
                        int_lmrks[42:48]))), (1,))
    
    cv2.fillConvexPoly(sense1, cv2.convexHull(
        np.concatenate((
                        int_lmrks[27:36],
                        int_lmrks[48:68]
                        ))), (1,))
    
    
    cv2.fillConvexPoly(sense2, cv2.convexHull(
        np.concatenate((int_lmrks[36:40],
                        #int_lmrks[17:22],
                        #int_lmrks[22:27],
                        int_lmrks[42:48],
                        int_lmrks[27:36],
                        int_lmrks[48:68]))), (1,))
    
    
    return  sense1 , sense2

def crop_img(img , mask):
    
    mask = np.squeeze(mask)
    mask = mask.astype('int32')
    
    res = np.zeros((img.shape), dtype = np.uint8)
    r_channel, g_channel, b_channel = cv2.split(img)
    
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if mask[i][j]==1:
                mask[i][j]=0
            else:
                mask[i][j]=255
            r_channel[i][j] = mask[i][j] & r_channel[i][j]
            g_channel[i][j] = mask[i][j] & g_channel[i][j]
            b_channel[i][j] = mask[i][j] & b_channel[i][j]
            
    res = cv2.merge([r_channel , g_channel , b_channel])
            
    return res
    

def single_image_crop(src , dst , write_or_show = 1):
    problem = []
    img_path = src
    # 'D:\FF++\data\train\Deepfakes\1\1.jpg'
    
    assert os.path.exists(img_path) 
    
    try:
        xy_array , xy_list ,img = find_landmark(img_path) #第一個mask 要用 , 第二個是畫landmark要用 , 第三個就是圖
    
        """  是否要畫上 landmarks ?"""
        for (x,y) in xy_list:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            
        show_img(img)
    except :
         print("skip and store into problem")
         problem.append(src)
         pass
# =============================================================================
#         mask1 , mask2= get_image_hull_mask( (img.shape[0],img.shape[1]) , xy_array )
#         res = crop_img(img, mask1)
#         
#         if write_or_show == 1:
#             cv2.imwrite( dst ,res)
#         else:
#             show_img(res)
#         
#         
#     except :
#         print("skip and store into problem")
#         problem.append(src)
#         pass
# 
#     return problem
# =============================================================================


def makelist(path1 , path2):
    l = []
    l2 = []
    for i in os.listdir(path1):
        
        pp2 = os.path.join(path2,i) 
        if not os.path.exists(pp2):
            os.makedirs(pp2)
        
        pp = os.path.join(path1,i) 
        if os.path.exists(pp):
            
            for r in range(1,51):
                gg = os.path.join(pp,str(r)+'.jpg') 
                l.append(gg)
                gg2 = os.path.join(pp2,str(r)+'.jpg') 
                l2.append(gg2)
        else:
            print(pp,"is not exists!!!!")
    return l , l2



if __name__=="__main__":
    
    count = 0
    
    src_list = []
    dst_list = []
    all_type = ['Deepfakes','Face2Face', 'FaceShifter' , 'FaceSwap' , 'NeuralTextures' ,'original']
    mask_type = ['Deepfakes_mask','Face2Face_mask', 'FaceShifter_mask' , 'FaceSwap_mask' , 'NeuralTextures_mask' , 'original_mask']
    
    problem = []
    

    #src = r'D:\FF++\data\train2\Deepfakes\001_870\1.jpg'
    
    
    imageList = [r'D:\FF++\data\train2\original\005\1.jpg' , r'D:\FF++\data\train2\original\001\1.jpg' ,
                 r'D:\FF++\data\train2\FaceShifter\001_870\1.jpg' , r'D:\FF++\data\train2\FaceShifter\005_010\1.jpg'
                 ]
    
    
    
    
    
    
    single_image_crop(imageList[3] , ' ', write_or_show = 0 ) # write = 1 , show = 0
    
# =============================================================================
#     for w in range(0,6):
#         f = r'D:\FF++\data\train2'
#         
#         f1 = os.path.join(f,all_type[w])
#         f2 = os.path.join(f,mask_type[w])
#         
#         src_list  , dst_list = makelist(f1,f2)
#         
#         for i in range(0,len(src_list)):
#             print(src_list[i])
#             temp = single_image_crop(src_list[i] , dst_list[i] )
#             
#             problem += temp
# 
#     
#     f = open('problem.txt' , 'w')
#     f.write(str(problem))
#     f.close()
# =============================================================================

























