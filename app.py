# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:53:34 2021

@author: Shiva
"""


import streamlit as st
import cv2
import gc
from PIL import Image 
import numpy as np
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model
from keras import backend as K
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
##os.chdir('Mask_RCNN')
#sys.path.append('Mask_RCNN')
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize
#%matplotlib inline
#https://www.tensorflow.org/lite/models/pose_estimation/overview

with open('D:\\Appliedaicourse\\Casestudy2\\data\\label_descriptions.json') as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

#combine categories for simplification
foot_wear = ['shoe']

upper_body_wear = ['vest','top, t-shirt, sweatshirt','sweater','sleeve','shirt, blouse','neckline','lapel','jacket','hood',
              'epaulette','collar','cardigan',]
lower_body_wear =['pocket', 'pants', 'shorts', 'skirt']
wholebody = ['cape', 'coat', 'dress', 'jumpsuit']



st.title('Fashion Recomendatation')
st.subheader("Upload file")


uploaded_file = st.file_uploader("choose an image.... ",type=['png','jpg'])
status_text = st.empty()
if uploaded_file is not None:
    #image = load_image(uploaded_file)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(512,512), interpolation=cv2.INTER_AREA)
    #st.image(image)
    status_text.text('ImageLoaded Done!')
class Module1:
    def __init__(self):
        model_path = "D:\Appliedaicourse\Casestudy2\posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors information from the model file
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def model_output(self,image):
        '''
        Input: path of the image
        Output: HeatMap, Offsets'''
        #template_image_src = cv2.imread(path)
        template_image = cv2.resize(image, (257, 257))
        template_input = np.expand_dims(template_image.copy(), axis=0)
        floating_model = self.input_details[0]['dtype'] == np.float32

        if floating_model:
            template_input = (np.float32(template_input) - 127.5) / 127.5
        # Process template image
        # Sets the value of the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], template_input)
        # Runs the computation
        self.interpreter.invoke()
        # Extract output data from the interpreter
        template_output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        template_offset_data = self.interpreter.get_tensor(self.output_details[1]['index'])
        # Getting rid of the extra dimension
        template_heatmaps = np.squeeze(template_output_data)
        template_offsets = np.squeeze(template_offset_data)
        #print("template_heatmaps' shape:", template_heatmaps.shape)
        #print("template_offsets' shape:", template_offsets.shape)
        return template_heatmaps,template_offsets
    def parse_output(self,heatmap_data,offset_data, threshold):
        '''
        Input:
        heatmap_data - hetmaps for an image. Three dimension array
        offset_data - offset vectors for an image. Three dimension array
        threshold - probability threshold for the keypoints. Scalar value
        Output:
        array with coordinates of the keypoints and flags for those that have
        low probability
        COnd: If nose and (any one eye) and (any one hip) and (any one ankle) is present then only Bool will be True
        '''

        joint_num = heatmap_data.shape[-1]
        pose_kps = np.zeros((joint_num,3), np.uint32)

        for i in range(heatmap_data.shape[-1]):
            joint_heatmap = heatmap_data[...,i]
            max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
            remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
            pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
            pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
            max_prob = np.max(joint_heatmap)

            if max_prob > threshold:
                if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
                    pose_kps[i,2] = 1
            bool = (pose_kps[0][2]==1) and ((pose_kps[1][2] or pose_kps[2][2]) ==1) and ((pose_kps[11][2] or pose_kps[12][2]) ==1) and ((pose_kps[15][2] or pose_kps[16][2]) ==1)
        return bool
    def final(self,image):
        '''
        Input: path of the Image 
        output: Bool true or flase 
        Explanation : it takes path and it sends to model to get the 
        output and it sends to parse output function to get wheater it is a full pose or not'''
        template_heatmaps,template_offsets = self.model_output(image)
        bool = self.parse_output(template_heatmaps,template_offsets,0.5)
        return bool

##module 2
class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = 46 + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    
config = FashionConfig()
@st.cache
class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir='\\D:\Appliedaicourse\Casestudy2\\')

model_path1 ='D:/Appliedaicourse/Casestudy2/mask_rcnn_fashion_0005-0.34882.h5'
assert model_path1 != '', "Provide path to trained weights"

#print("Loading weights from ", model_path1)
model.load_weights(model_path1, by_name=True)#load trained model

class Module2:
  
 
    def get_bbox(self,image):
        '''input: Takes the  image
        output: it detects the mask,bboxes and 
        also it send the dictionary contains with class id,score,masks,boxes'''

        result = model.detect([image])
    
        if result[0]['masks'].size > 0:
            mask = np.zeros((image.shape[0], image.shape[1], result[0]['masks'].shape[-1]), dtype=np.uint8)
            for m in range(result[0]['masks'].shape[-1]):
                mask[:, :, m] = cv2.resize(result[0]['masks'][:, :, m].astype('uint8'), 
                                        (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            y = image.shape[0]/512
            x = image.shape[1]/512
            boxes = (result[0]['rois'] * [y, x, y, x]).astype(int)
        
        else:
            mask, boxes = result[0]['masks'], result[0]['boxes']
        visualize.display_instances(image, boxes, mask, result[0]['class_ids'], ['bg']+label_names, result[0]['scores'],
                                 figsize=(8, 8))
        return image,mask,boxes,result
    def get_bounding_box_for_parts(self,img,boxes,result,class_names):
        '''Input : image,boxes result from model,class_names list
        Output : returns a tuple of 3 list contain foot,upper_wear,lower_body wear'''
    
        #boxes = result[0]['rois']
        class_id = result[0]['class_ids']
        scores = result[0]['scores']
        masks = result[0]['masks']
        height , width = img.shape[:2]
        foot  = []
        upper_body = []
        lower_body = []
        whole_bod = []
    
        bool =  result[0]['rois'].shape[0] == result[0]['masks'].shape[-1] == result[0]['class_ids'].shape[0]
        if bool:
            if boxes is not None:
                for i in range(len(boxes)):
                    y1,x1,y2,x2 = boxes[i]
                    label  = class_names[class_id[i]-1]
                    score_got  = scores[i] if scores is not None else 0
                    if score_got >0.70:
                        if label in foot_wear:
                            foot.append((y1,x1,y2,x2))
                        if label in upper_body_wear:
                            upper_body.append((y1,x1,y2,x2))
                        if label in lower_body_wear:
                            lower_body.append((y1,x1,y2,x2))
                        if label in wholebody:
                            whole_bod.append((y1,x1,y2,x2))
        return upper_body,lower_body,foot
    ## Getting only the objects from the image
    def crop(self,y1,x1,y2,x2,image):
        '''Input : co-ordinates,image
           It will crop the image with the given co-ordinates
           output: croped image'''
    
        output = image[int(y1):int(y2), int(x1):int(x2)]
        return output

    def get_croped_images(self,upper_wear,lower_wear,foot_wear,img):
        
        '''Input :bounding_box of upper_wear,lower_wear,foot_wear
           Function -> This function will take the boundign box of upperwears like sleeve shirt pocket in this it will combine
           all the bounding box parts of the particular portion and return the images of particular wear.
           output : It returns the Upper_wear,Lower_wear,Foot wear Images'''
        ##getting full upper_body cropping
        if all([len(lower_wear)!=0, len(upper_wear)!=0 , len(foot_wear)!=0]):
        
            up_y1,up_x1 = np.min(np.array(upper_wear),axis=0)[:2]
            up_y2,up_x2 = np.max(np.array(upper_wear),axis=0)[2:]
            upper_image = self.crop(up_y1,up_x1,up_y2,up_x2,img)
            #Getting LowerImage
            L_y1,L_x1 =   np.min(np.array(lower_wear),axis=0)[:2]
            L_y2,L_x2 =   np.max(np.array(lower_wear),axis=0)[2:]
            lower_image = self.crop(L_y1,L_x1,L_y2,L_x2,img)
            ## Getting foot_wear
            F_y1,F_x1   = np.min(np.array(foot_wear),axis=0)[:2]
            F_y2,F_x2   = np.max(np.array(foot_wear),axis=0)[2:]
            foot_image  = self.crop(F_y1,F_x1,F_y2,F_x2,img)
            return upper_image,lower_image,foot_image
        else:
            print('Missing parts')
            return None
    def plot_wears(self,image):
        '''Input : image
        Function -> This function will plot images merging with upper_wear, lower_wear and Foot wear predicted from the model
        Output : Image of Upperwear, LowerWear and Footwear'''
        start = time.time()
        
        image,mask,boxes,result = self.get_bbox(image)
        a,b,c = self.get_bounding_box_for_parts(image,boxes,result,label_names)
        temp = self.get_croped_images(a,b,c,image)
        if temp!=None:
            '''for i in temp:
                plt.imshow(i)
                plt.show()
                print(i.shape)'''
            return temp[0],temp[1],temp[2]
        else:
            print("Cant be Done")
        end = time.time()
        print("Time Taken: {}".format(end-start))
        print('\n')
        return temp[0],temp[1],temp[2]

        
    
class Module3:
    def __init__(self):
        
        self.model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=None, 
                                                       input_shape=(512,512,3),pooling='max')
    def extract_features(self,img):
        '''Input :  Image and pretrained densenet12 model 
           output : Image Embedings of size 1024 for an Image'''
        preprocessed_img = preprocess_input(np.expand_dims(cv2.resize(img,(512,512)),axis=0))
        #print(type(norm(preprocessed_img)))
        features = self.model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features
@st.cache
class Module4:
    ''' This module will give the path of similar images by loading 
    pretrained models of nearest neighbour of particular upper or lower or foot wear'''
    @st.cache
    def __init__(self):

        self.upmodel = pickle.load(open('D:\Appliedaicourse/Casestudy2/UpperModel.pkl','rb'))
        self.lomodel = pickle.load(open('D:\Appliedaicourse\Casestudy2\LowerModel.pkl','rb'))
        self.fomodel = pickle.load(open('D:\Appliedaicourse\Casestudy2\FootModel.pkl','rb'))
        self.upperwear=pickle.load(open(r'D:\Appliedaicourse\Casestudy2\UpperWear.pkl','rb'))
        self.lowerwear=pickle.load(open(r'D:\Appliedaicourse\Casestudy2\lowerwear.pkl','rb'))
        self.footwear= pickle.load(open(r'D:\Appliedaicourse\Casestudy2\FootWear.pkl','rb'))
    @st.cache
    def get_images(self,upper,lower,foot):
        '''Input : Embedings of lower,upper,foot 
           output : It gives the path of similar_products'''
        distances1, indices1 = self.upmodel.kneighbors([upper])
        distances2, indices2 = self.lomodel.kneighbors([lower])
        distances3, indices3 = self.fomodel.kneighbors([foot])
        upper_paths = self.upperwear['Path'][indices1[0]]
        lower_paths = self.lowerwear['Path'][indices1[0]]
        foot_paths  = self.footwear['Path'][indices1[0]]
        
        return upper_paths,lower_paths,foot_paths
        


m1 = Module1()
m2 = Module2()
m3 = Module3()
m4 = Module4()
s = time.time()
if m1.final(image)==True:
    
    st.header("Given Image is Full Front Pose")
    st.subheader('Sending to Localization and article detection')
    a,b,c = m2.plot_wears(image)
    st.subheader('seperating Lower wear, Upper Wear and Foot Wear')
    st.image([a,b,c],caption=['Upper_wear','Lower_wear','Foot_wear'])
    a1,b1,c1 = m3.extract_features(img = a),m3.extract_features(img = b),m3.extract_features(img = c)
    upper_paths,lower_paths,foot_paths = m4.get_images(a1,b1,c1)
    upper = []
    lower = []
    foot  = []
    for i in upper_paths[:50]:
        image = cv2.imread('D:\Appliedaicourse\Casestudy2\\data\\'+(i).replace('/','\\'))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(130,130),interpolation=cv2.INTER_AREA)
        upper.append(image)
    for i in lower_paths[:50]:
        image = cv2.imread('D:\Appliedaicourse\Casestudy2\\data\\'+(i).replace('/','\\'))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(130,130),interpolation=cv2.INTER_AREA)
        lower.append(image)
    for i in foot_paths[:50]:
        image = cv2.imread('D:\Appliedaicourse\Casestudy2\\data\\'+(i).replace('/','\\'))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(130,130),interpolation=cv2.INTER_AREA)
        foot.append(image)
    st.header('Upper wear')
    st.image(a)
    st.subheader("Recomendataion for Upper wear")
    st.image(upper)
    st.header('Lower wear')
    st.image(b)
    st.subheader('Recomendataion for Lower Wear' )
    st.image(lower)
    st.header('Foot wear')
    st.image(c)
    st.subheader('Recomendatation for Foot Wear')
    st.image(foot)
else:
    st.header('Please Upload Full Front Pose Image')
st.text('The time taken for this recomedation :')
st.text(time.time()-s)
gc.collect()
