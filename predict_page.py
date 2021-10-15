#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import numpy as np
from PIL import Image
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import SessionState
def show_predict_page():
    
    class ConvNet(nn.Module):
        def __init__(self, num_classes):
            super(ConvNet, self).__init__()

            self.conv1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding = 2),
                                       nn.BatchNorm2d(6),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2, stride=2))

            self.conv2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=3, padding = 2),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding = 1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2, stride = 2))

            self.output = nn.Linear(2048, num_classes)

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)


            out = out.reshape(-1,2048)
            #print(out.shape)
            out = self.output(out)
            #print(out.shape)
            return F.softmax(out, dim=1)

  

    alpha=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    #my_model=torch.load('model.pt')
    model = ConvNet(26)
    model.load_state_dict(torch.load('your project'))
    model.eval()


       

     


       

    
    
    
    #########################################################
        


    ss=SessionState.get(li= [])
    def display_text(bounds):
        text = []
        for x in bounds:
            
            #t = x[1]
            text.append(x)
        text = ' '.join(text)
        return text 

    listo=[]
    #st.sidebar.title('Language Selection Menu')
#     st.sidebar.subheader('Select...')
#     src = st.sidebar.selectbox("rom Languag",['hand_g','nglish'])

#     st.sidebar.subheader('Select...')
#     destination = st.sidebar.selectbox("o Language",['nglish','hand_g'])

#     st.sidebar.subheader("Enter Text")
#     area = st.sidebar.text_area("Auto Detection Enabled","")

#     helper = {'hand_g':'hg','nglish':'en'}
#     dst = helper[destination]
#     source = helper[src]

#     if st.sidebar.button("Translate!"):
#         if len(area)!=0:
#             sour = translator.detect(area).lang
#             answer = translator.translate(area, src=f'{sour}', dest=f'{dst}').text
#             #st.sidebar.text('Answer')
#             st.sidebar.text_area("Answer",answer)
#             st.balloons()
#         else:
#             st.sidebar.subheader('Enter Text!')    


    #st.set_option('deprecation.showfileUploaderEncoding',False)
    st.title('Hands gesture detection')
    #st.subheader('Optical Character Recognition with Voice output')
    st.text('Chose an Image/s for detection.')

    image_file = st.file_uploader("Upload Image")#,accept_multiple_files=True)

    #st.button("add_space",key=str(np.random.rand()))
    if st.button("Convert"):

        if image_file is not None:
#             img = Image.open(image_file)
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            #this is your image
            img = opencv_image

            # pre processing the image and prepare it for the classification
            img = cv2.resize(img, (64,64))
            img_tensor = torch.from_numpy(img[:,:,1:2])
            img_tensor=img_tensor.reshape(1,1,64,64)


            st.subheader('Image you Uploaded...')
 

            pred= model(img_tensor*1.1)

            top_p_pred, top_class_pred = pred.topk(1, dim=1) 
            #st.write(alpha[top_class_pred.item()])
            ss.li.append(alpha[top_class_pred.item()])
    if st.button("aadd_space"):
        ss.li.append(" ")
            #ss.li.append(listo)
    if st.button("Text"):
        st.write( ss.li)

    st.write("".join(ss.li))

