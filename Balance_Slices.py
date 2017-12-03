import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
#import argparse

def jugaad(app_img,num):
    rnd_values = []
    rnd_values = [np.random.randint(1,101) for val in range(num)]
    for l in rnd_values:
        x = app_img[l]
        itr = 0
        for batch in datagen.flow(x.reshape(1, x.shape[0], x.shape[1], x.shape[2]), batch_size=1):
            itr += 1
            if(itr > 1):
                break
            app_img.append(batch.reshape(x.shape[0], x.shape[1], x.shape[2]))
    return app_img

if __name__ == "__main__":
     
    #if not os.path.exists(directory):
    #os.makedirs(directory)

    # Importing images directly pre-processed by D-O-DHAVAL-G
    DATA_DIRECTORY = 'D://Data Science//DS Bowl//Extracted_Images//'
    SAVE_DIRECTORY = 'D://Data Science//DS Bowl//JugaadImages//'
    patients = [patient for patient in os.listdir(DATA_DIRECTORY)]
    
    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
 
    slices = []
    num_slices = []
    
    data = []
    for patient in patients:
        print(patient)
        cnt_iteration = 0
        for imgs in os.listdir(DATA_DIRECTORY+patient):
            if(cnt_iteration%2==0):
                img = cv2.imread(DATA_DIRECTORY+patient+"//"+imgs,0)
                img = cv2.resize(img,(320,320))
                backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                slices.append(np.array(backtorgb))
            cnt_iteration+=1
        data.append(slices)
        slices = []
        
    num_slices = [len(data[i]) for i in range(len(data))]
    print("")
    print("Number of slices per patient:")
    print(num_slices)
    
    max_slices = max(num_slices)
    left_out = [(max_slices-num) for num in num_slices]
    print("")
    print("Number of Slices to append for Each Patient!")
    print(left_out)
    
    new_data = data[:]
    print("")
    
    for patient in patients:
        id = patients.index(patient)
        data_temp = new_data[id]
        num = left_out[id]
        appendData = jugaad(data_temp,num)
        data[id] = appendData
        print("Patient %d Data balanced!" %(id))
        data_temp = []
    
    print("")
    print("Number of patients in the dataset: ", len(data))
    print("")
    print("Number of slices in each patient:")
    print("")
    for cnt in range(len(data)):
        print("Number of slices in patient %d is %d" %(cnt, len(data[cnt])))
    
    ("Ladies & Gentlemen! Signing off! ^_^ ")