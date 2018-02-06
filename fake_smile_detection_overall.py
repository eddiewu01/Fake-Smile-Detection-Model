#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:57:20 2017

@author: edwardwu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import sys
import shutil

############# Related folders for saving images/results ###########
folder_fake = '/Users/edwardwu/Desktop/CIS519_Project/fake_smiles'
folder_real = '/Users/edwardwu/Desktop/CIS519_Project/true_smiles'

folder_fake_faces = '/Users/edwardwu/Desktop/CIS519_Project/fake_faces'
folder_real_faces = '/Users/edwardwu/Desktop/CIS519_Project/true_faces'

folder_fake_mouths = '/Users/edwardwu/Desktop/CIS519_Project/fake_mouths'
folder_real_mouths = '/Users/edwardwu/Desktop/CIS519_Project/true_mouths'

folder_fake_eyes = '/Users/edwardwu/Desktop/CIS519_Project/fake_eyes'
folder_real_eyes = '/Users/edwardwu/Desktop/CIS519_Project/true_eyes'

folder_results_fake_test_faces  = '/Users/edwardwu/Desktop/CIS519_Project/result/fake_test_fromfaces'
folder_results_real_test_faces  = '/Users/edwardwu/Desktop/CIS519_Project/result/true_test_fromfaces'

folder_results_fake_test_eyes  = '/Users/edwardwu/Desktop/CIS519_Project/result/fake_test_fromeyes'
folder_results_real_test_eyes  = '/Users/edwardwu/Desktop/CIS519_Project/result/true_test_fromeyes'

folder_results_fake_test_mouths  = '/Users/edwardwu/Desktop/CIS519_Project/result/fake_test_frommouths'
folder_results_real_test_mouths  = '/Users/edwardwu/Desktop/CIS519_Project/result/true_test_frommouths'


folder_results_fake_test_overall = '/Users/edwardwu/Desktop/CIS519_Project/result/fake_test_overall'
folder_results_real_test_overall = '/Users/edwardwu/Desktop/CIS519_Project/result/true_test_overall'


def load_images_from_folder(folder):
    
    ''' 
    Input:
        folder             :The path of the image folder.
    Output:
        images             :A (N,) numpy array, where N is the total number of 
                            images.
    '''
    
    images = []
    for filename in sorted(os.listdir(folder)):
        #print filename
        if filename == ".DS_Store":
            continue
        #print filename
        img = Image.open(os.path.join(folder,filename)).convert('L')            # open the image and convert to uint8 (gray scale)
        img = np.array(img)                                                     # convert image to a numpy array
        if img is not None:
            images.append(img)
    images = np.array(images)
    return images

def resize_image(images, size):
    
    '''
    Input: 
        images             :An object of shape (N,). N is the total number of 
                            images.
        size               :A ndarray (rows, cols) of the output image shape.
    Output:
        resized_images     :An (N,rows,cols) numpy array.
    '''
    
    resized_images = []
    for i,j in enumerate(images):
        resized_images.append(resize(images[i], ((size[0]),size[1]),            # resize the image
                                     mode='reflect'))
    resized_images = np.array(resized_images)                                   # convert image to a numpy array
    return resized_images

def detectFace(img):
    
    '''
    Input:
        img                 : A gray scale image represented by numpy array.
    Output:
        bbox                : The four corners of bounding boxes for all 
                              detected faces in numpy arrray of shape (number 
                              of detected faces,4,2).
                                ++++++++++++++++++
                                +(x0,y0)  (x1,y1)+
                                +                +
                                +                +
                                +      bbox      +
                                +                +
                                +                +
                                +(x2,y2)  (x3,y3)+
                                ++++++++++++++++++
    '''
    
    face_cascade = cv2.CascadeClassifier('/Users/edwardwu/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    delta_scal = 1.5
    while True:
        faces = face_cascade.detectMultiScale(img, 3.0-delta_scal, 2)
        if len(faces) != 0:                                                     # at least one face is detected
            break
        else:                                                                   # no face detected, re-detecting with new parameters...
            if 3.0-delta_scal > 1.01:
                delta_scal += 0.01
            else:
                break
    bbox = np.zeros([len(faces),4,2])
        
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)                          # draw a red rectangle around the face  
        bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])
        
    if bbox.shape[0] > 1:
        print("\nWarning! Multiple faces detected! Image shown below...")
        #plt.figure()
        #plt.imshow(img, cmap='gray')
    
    if bbox.shape[0] == 0:
        print("\nWarning! No faces detected! Image shown below...")
        #plt.figure()
        #plt.imshow(img, cmap = 'gray')
        faces = face_cascade.detectMultiScale(img, 1, 1)
        for i, (x,y,w,h) in enumerate(faces):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)                          # draw a red rectangle around the face  
            bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])
    return bbox[0, :, :].astype(int)  
    
def detectMouth(img, orig_img):
    mouth_cascade = cv2.CascadeClassifier('/Users/edwardwu/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
    mouths = mouth_cascade.detectMultiScale(img)
    
    
    bbox = np.zeros([len(mouths),4,2])
    for i, (x,y,w,h) in enumerate(mouths):
        y = int(y - 0.15*h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)                          # draw a red rectangle around the face  
        bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])

    if bbox.shape[0] == 0: # if no mouth is detected 
        temp = np.zeros([4,2])
        temp[0] = np.array([230,90])
        temp[1] = np.array([230,175])
        temp[2] = np.array([260,90])
        temp[3] = np.array([260,175])
        return temp.astype(int) # returning a rectangle which we think should contain mouth for most images (four sets of x,y coordinates above)
    elif bbox.shape[0] > 3: # if too many mouths detected
        choose = bbox.shape[0]/2
        #choose = bbox.shape[0]-1
        return bbox[choose,:,:].astype(int) # returning the middle one
    else: # if only a few mouths detected
        return bbox[0,:,:].astype(int) # returning the first one
    
def detectEyes(img, orig_img):
    eye_cascade = cv2.CascadeClassifier('/Users/edwardwu/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(img, 1.1, 1)
    
    bbox = np.zeros([len(eyes),4,2])
    for i, (x,y,w,h) in enumerate(eyes):
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)                          # draw a red rectangle around the face  
        bbox[i,:,:] = np.array([[y,x],[y,x+w],[y+h,x],[y+h,x+w]])
    
    
    if bbox.shape[0] == 0: # if no eyes are detected
        temp = np.zeros([4,2])
        temp[0] = np.array([80,70])
        temp[1] = np.array([80,130])
        temp[2] = np.array([130,70])
        temp[3] = np.array([130,130])
        return temp.astype(int) # returning a rectangle which we think should contain an eye for most images
        
    elif bbox.shape[0] > 3: # if too many eyes detected
        choose = bbox.shape[0]/2
        #choose = bbox.shape[0]-2
        return bbox[choose,:,:].astype(int) # returning the middle one
    else: # if only a few eyes are detected
        return bbox[0,:,:].astype(int) # returning the first one
    
new_size = [300,300]

############## loading in original images and storing them for evaluation in the end ########3 
orig_face_images = load_images_from_folder(folder_fake)                         # load images from folder
orig_face_X_fake = resize_image(orig_face_images,new_size)                                    # resize to 300*300
orig_y_fake = np.ones([orig_face_X_fake.shape[0]])                                           # fake smile labels: 1  

orig_face_images = load_images_from_folder(folder_real)                         # load images from folder
orig_face_X_real = resize_image(orig_face_images,new_size) 
orig_y_real = np.zeros([orig_face_X_real.shape[0]]) 


X_raw = np.r_[orig_face_X_fake, orig_face_X_real]                                               # concat fake and real smile samples together
y_raw = np.r_[orig_y_fake, orig_y_real]                                                   # 1: fake, 0: real
indices = np.arange(X_raw.shape[0])
training_x, testing_x, training_y, testing_y, idx1, idx2 = train_test_split(X_raw, y_raw, indices, 
                                                    test_size=0.3, random_state=42)  # use random_state=42 to make sure each split selects the same set of rows (images)
X_raw_orig = np.r_[orig_face_X_fake, orig_face_X_real]
orig_X_test = X_raw_orig[idx2] #used for final evaluation
y_raw_orig = np.r_[orig_y_fake, orig_y_real]
orig_y_test = y_raw_orig[idx2]  # used for final evaluation

################ Working with original (uncropped) images ################
print("Loading original images..")
images = load_images_from_folder(folder_fake)                               # load images from folder
X_fake = resize_image(images,new_size)                                      # resize to 300*300
    
images = load_images_from_folder(folder_real)
X_real = resize_image(images,new_size)
    
X_fake_f = np.zeros([X_fake.shape[0],new_size[0],new_size[1]])
X_real_f = np.zeros([X_real.shape[0],new_size[0],new_size[1]])

toolbar_width = 1                                                           # not important, just for printing purpose...
sys.stdout.write("%s\r" % (" " * toolbar_width))                            # not important, just for printing purpose...
sys.stdout.flush()                                                          # not important, just for printing purpose...
sys.stdout.write("\b" * (toolbar_width+1))                                  # not important, just for printing purpose...
    
shutil.rmtree(folder_fake_faces, ignore_errors=True)                        # remove the folder if it already exists
os.makedirs(folder_fake_faces)                                              # create a new folder
shutil.rmtree(folder_real_faces, ignore_errors=True)                        # remove the folder if it already exists
os.makedirs(folder_real_faces)                                              # create a new folder
    
print X_fake.shape
for i in range(X_fake.shape[0]):
    bbox = detectFace((255*X_fake[i,:,:]).astype('uint8'))                  # detects the face of the original image
    face_i = X_fake[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
    X_fake_f[i,:,:] = resize(face_i, (new_size[0],new_size[1]),
            mode='reflect')                                # resize to 300*300
    plt.imsave(folder_fake_faces+"/fake_smile_"+str(i)+".jpg",
               X_fake_f[i,:,:], cmap='gray')                                # save the detected face
    sys.stdout.write("\rDetecting and saving fake smile faces...%.1f%%" % 
                     (i*100/(X_fake.shape[0]-1)))                           # not important, just for printing purpose...
    sys.stdout.flush()                                                      # not important, just for printing purpose...
        
sys.stdout.write("\n")
        
for i in range(X_real.shape[0]):
    bbox = detectFace((255*X_real[i,:,:]).astype('uint8'))                  # detects the face of the original image
    face_i = X_real[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
    X_real_f[i,:,:] = resize(face_i, (new_size[0],new_size[1]), 
            mode='reflect')                                # resize to 300*300
    plt.imsave(folder_real_faces+"/real_smile"+str(i)+".jpg",
               X_real_f[i,:,:], cmap='gray')                                # save the detected face
        
    sys.stdout.write("\rDetecting and saving real smile faces...%.1f%%" % 
                         (i*100/(X_real.shape[0]-1)))                           # not important, just for printing purpose...
    sys.stdout.flush()                                                      # not important, just for printing purpose...
    
sys.stdout.write("\n")   

y_fake_f = np.ones([X_fake_f.shape[0]])                                           # fake smile labels: 1
y_real_f = np.zeros([X_real_f.shape[0]])  


################ Working with cropped face images for eyes detection ################
print("Loading previously cropped face images..")
images = load_images_from_folder(folder_fake_faces)                               # load images from folder
X_fake = resize_image(images,new_size)                                      # resize to 300*300

images = load_images_from_folder(folder_real_faces)
X_real = resize_image(images,new_size)

X_fake_e = np.zeros([X_fake.shape[0],new_size[0],new_size[1]])
X_real_e = np.zeros([X_real.shape[0],new_size[0],new_size[1]])

toolbar_width = 1                                                           # not important, just for printing purpose...
sys.stdout.write("%s\r" % (" " * toolbar_width))                            # not important, just for printing purpose...
sys.stdout.flush()                                                          # not important, just for printing purpose...
sys.stdout.write("\b" * (toolbar_width+1))                                  # not important, just for printing purpose...

shutil.rmtree(folder_fake_eyes, ignore_errors=True)                        # remove the folder if it already exists
os.makedirs(folder_fake_eyes)                                              # create a new folder
shutil.rmtree(folder_real_eyes, ignore_errors=True)                        # remove the folder if it already exists
os.makedirs(folder_real_eyes)                                              # create a new folder

for i in range(X_fake.shape[0]):
    bbox = detectEyes((255*X_fake[i,:,:]).astype('uint8'), orig_face_X_fake[i,:,:])                  # detects the face of the original image
    face_i = X_fake[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
    X_fake_e[i,:,:] = resize(face_i, (new_size[0],new_size[1]),
                             mode='reflect')                                # resize to 300*300
    plt.imsave(folder_fake_eyes+"/fake_smile_eyes"+str(i)+".jpg",
               X_fake_e[i,:,:], cmap='gray')                                # save the detected face
    
    sys.stdout.write("\rDetecting and saving fake smile eyes...%.1f%%" % 
                     (i*100/(X_fake.shape[0]-1)))                           # not important, just for printing purpose...
    sys.stdout.flush()                                                      # not important, just for printing purpose...
    
sys.stdout.write("\n")

    
for i in range(X_real.shape[0]):
    bbox = detectEyes((255*X_real[i,:,:]).astype('uint8'), orig_face_X_real[i,:,:])                  # detects the face of the original image
    face_i = X_real[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
    X_real_e[i,:,:] = resize(face_i, (new_size[0],new_size[1]), 
                             mode='reflect')                                # resize to 300*300
    plt.imsave(folder_real_eyes+"/real_smile_eyes"+str(i)+".jpg",
               X_real_e[i,:,:], cmap='gray')                                # save the detected face
    
    sys.stdout.write("\rDetecting and saving real smile eyes...%.1f%%" % 
                     (i*100/(X_real.shape[0]-1)))                           # not important, just for printing purpose...
    sys.stdout.flush()                                                      # not important, just for printing purpose...

sys.stdout.write("\n")        

y_fake_e = np.ones([X_fake_e.shape[0]])                                           # fake smile labels: 1
y_real_e = np.zeros([X_real_e.shape[0]])                                          # real smile labels: 0


################ Working with cropped face images for mouth detection ################
print("Loading previously cropped face images..")
images = load_images_from_folder(folder_fake_faces)                               # load images from folder
X_fake = resize_image(images,new_size)                                      # resize to 300*300

images = load_images_from_folder(folder_real_faces)
X_real = resize_image(images,new_size)

X_fake_m = np.zeros([X_fake.shape[0],new_size[0],new_size[1]])
X_real_m = np.zeros([X_real.shape[0],new_size[0],new_size[1]])

toolbar_width = 1                                                           # not important, just for printing purpose...
sys.stdout.write("%s\r" % (" " * toolbar_width))                            # not important, just for printing purpose...
sys.stdout.flush()                                                          # not important, just for printing purpose...
sys.stdout.write("\b" * (toolbar_width+1))                                  # not important, just for printing purpose...

shutil.rmtree(folder_fake_mouths, ignore_errors=True)                        # remove the folder if it already exists
os.makedirs(folder_fake_mouths)                                              # create a new folder
shutil.rmtree(folder_real_mouths, ignore_errors=True)                        # remove the folder if it already exists
os.makedirs(folder_real_mouths)                                              # create a new folder

for i in range(X_fake.shape[0]):
    bbox = detectMouth((255*X_fake[i,:,:]).astype('uint8'), orig_face_X_fake[i,:,:])                  # detects the face of the original image
    face_i = X_fake[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
    X_fake_m[i,:,:] = resize(face_i, (new_size[0],new_size[1]),
                             mode='reflect')                                # resize to 300*300
    plt.imsave(folder_fake_mouths+"/fake_smile_mouth"+str(i)+".jpg",
               X_fake_m[i,:,:], cmap='gray')                                # save the detected face
    
    sys.stdout.write("\rDetecting and saving fake smile mouths...%.1f%%" % 
                     (i*100/(X_fake.shape[0]-1)))                           # not important, just for printing purpose...
    sys.stdout.flush()                                                      # not important, just for printing purpose...
    
sys.stdout.write("\n")
    
for i in range(X_real.shape[0]):
    bbox = detectMouth((255*X_real[i,:,:]).astype('uint8'), orig_face_X_real[i,:,:])                  # detects the face of the original image
    face_i = X_real[i,bbox[0,0]:bbox[2,0],bbox[0,1]:bbox[1,1]]              # crop out the face from the original image
    '''
    try:
        X_real_m[i,:,:] = resize(face_i, (new_size[0],new_size[1]), 
                             mode='reflect')                                # resize to 300*300
    except ValueError:
        pass
        '''
    X_real_m[i,:,:] = resize(face_i, (new_size[0],new_size[1]), 
                             mode='reflect')     
    plt.imsave(folder_real_mouths+"/real_smile_mouth"+str(i)+".jpg",
               X_real_m[i,:,:], cmap='gray')                                # save the detected face
    
    sys.stdout.write("\rDetecting and saving real smile mouths...%.1f%%" % 
                     (i*100/(X_real.shape[0]-1)))                           # not important, just for printing purpose...
    sys.stdout.flush()                                                      # not important, just for printing purpose...

sys.stdout.write("\n")                                                      # not important, just for printing purpose...
    
y_fake_m = np.ones([X_fake_m.shape[0]])                                           # fake smile labels: 1
y_real_m = np.zeros([X_real_m.shape[0]]) 




############## Setting up tensorflow for CNN training ###############
def reset_graph(seed=42):
    tf.reset_default_graph()
#    tf.set_random_seed(seed)
#    np.random.seed(seed)                                                       # uncomment if want to make train test split return the same samples every time

height = new_size[0]
width = new_size[1]
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_fmaps = 16
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_fmaps = 50
conv2_ksize = 3
conv2_stride = 3
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 2

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, 
                         kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 2500])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                              labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# defined function for tf_cnn training    
def tf_cnn_training(X_train, y_train, X_test, y_test, num_of_batches, n_epochs,
                    folder_results_fake, folder_results_real, original_testing_img):
    with tf.Session() as sess:
        init.run()
        accuracies = []
        for epoch in range(n_epochs):
            acc_train = 0
            acc_test = 0
            
            for batch in range(num_of_batches):                                     # feed in the training data one batch at a time
                from_i = batch*batch_size
                to_i = (batch+1)*batch_size
                
                if batch != num_of_batches-1:                                       # not last batch?
                    sess.run(training_op, feed_dict={X: X_train[from_i:to_i], 
                                                     y: y_train[from_i:to_i]})
                else:                                                               # last batch
                    sess.run(training_op, feed_dict={X: X_train[from_i:], 
                                                     y: y_train[from_i:]})
                    
                acc_train += accuracy.eval(feed_dict={X: X_train[from_i:to_i], 
                                         y: y_train[from_i:to_i]}) / num_of_batches
                acc_test  += accuracy.eval(feed_dict={X: X_test, 
                                                      y: y_test}) / num_of_batches
                
            pred_fake_test = sess.run(Y_proba, feed_dict={X: X_test, y: y_test})    # get the output (Y_proba) of the model
            pred_fake_indices_test = np.where(pred_fake_test[:,1]>=0.5)             # get the indices of the samples that is predicted "fake". If Y_proba >= 0.5, then it is predicted "fake"
            pred_real_indices_test = np.where(pred_fake_test[:,1]<0.5)              # get the indices of the samples that is predicted "real".if Y_proba < 0.5, then it is predicted "real"
            accuracies.append(acc_test)
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
             
        print("Saving the results...")
        shutil.rmtree(folder_results_fake, ignore_errors=True)                 # remove the folder if it already exists
        os.makedirs(folder_results_fake)                                       # create a new folder
        shutil.rmtree(folder_results_real, ignore_errors=True)                 # remove the folder if it already exists
        os.makedirs(folder_results_real)                                       # create a new folder
        
        print(np.mean(accuracies))
        
        for i,j in np.ndenumerate(pred_fake_indices_test):                          # save all the predicted "fake" samples into the folder
            plt.imsave(folder_results_fake+"/fake_smile_pred_test_"+str(i[1])
                                                +".jpg",original_testing_img[j,:,:], cmap='gray')
        
        #print pred_real_indices_test
        #print len(pred_real_indices_test)
        for i,j in np.ndenumerate(pred_real_indices_test):                          # save all the predicted "real" samples into the folder
            plt.imsave(folder_results_real+"/real_smile_pred_test_"+str(i[1])
                                                +".jpg",original_testing_img[j,:,:], cmap='gray')
    
        return pred_fake_indices_test, pred_real_indices_test


  
################ training + testing using the faces ##############
print "Face detection predictions..."
X_raw = np.r_[X_fake_f, X_real_f]                                               # concat fake and real smile samples together
y_raw = np.r_[y_fake_f, y_real_f]                                                   # 1: fake, 0: real
indices = np.arange(X_raw.shape[0])
#print X_raw.shape, y_raw.shape  
training_x, testing_x, training_y, testing_y, idx1, idx2 = train_test_split(X_raw, y_raw, indices, 
                                                    test_size=0.3, random_state=42)
X_raw_orig1 = np.r_[orig_face_X_fake, orig_face_X_real]
orig_X_test1 = X_raw_orig1[idx2]
#y_raw_orig1 = np.r_[orig_y_fake, orig_y_real] 
#orig_y_test1 = y_raw_orig1[idx2]  
epochs = 10                                                                    # number of epochs to train this model. The larger the better
batches = 5                                                              # split the training data into batches to avoid insufficient memory error
batch_size = int(training_x.shape[0] / batches)  
face_pred_fake, face_pred_real = tf_cnn_training(training_x, training_y, testing_x, testing_y, batches, epochs,
                    folder_results_fake_test_faces, folder_results_real_test_faces, orig_X_test1)


############ training + testing using the eyes ##############
print "Eye detection predictions..."
X_raw = np.r_[X_fake_e, X_real_e]                                               # concat fake and real smile samples together
y_raw = np.r_[y_fake_e, y_real_e]                                                   # 1: fake, 0: real
indices = np.arange(X_raw.shape[0])
#print X_raw.shape, y_raw.shape  
training_x, testing_x, training_y, testing_y, idx1, idx2 = train_test_split(X_raw, y_raw, indices, 
                                                    test_size=0.3, random_state=42)
X_raw_orig2 = np.r_[orig_face_X_fake, orig_face_X_real]
orig_X_test2 = X_raw_orig2[idx2]
#y_raw_orig2 = np.r_[orig_y_fake, orig_y_real]
#orig_y_test2 = y_raw_orig2[idx2]  
epochs = 10                                                                    # number of epochs to train this model. The larger the better
batches = 5                                                              # split the training data into batches to avoid insufficient memory error
batch_size = int(training_x.shape[0] / batches)  

eye_pred_fake, eye_pred_real = tf_cnn_training(training_x, training_y, testing_x, testing_y, batches, epochs,
                    folder_results_fake_test_eyes, folder_results_real_test_eyes, orig_X_test2)


############ training + testing using the mouths ##############
print "Mouth detection predictions..."
X_raw = np.r_[X_fake_m, X_real_m]                                               # concat fake and real smile samples together
y_raw = np.r_[y_fake_m, y_real_m]                                                   # 1: fake, 0: real
indices = np.arange(X_raw.shape[0])
#print X_raw.shape, y_raw.shape  
training_x, testing_x, training_y, testing_y, idx1, idx2 = train_test_split(X_raw, y_raw, indices, 
                                                    test_size=0.3, random_state=42)
X_raw_orig3 = np.r_[orig_face_X_fake, orig_face_X_real]
orig_X_test3 = X_raw_orig3[idx2]
#y_raw_orig3 = np.r_[orig_y_fake, orig_y_real]
#orig_y_test3 = y_raw_orig3[idx2]   
epochs = 10                                                                    # number of epochs to train this model. The larger the better
batches = 5                                                              # split the training data into batches to avoid insufficient memory error
batch_size = int(training_x.shape[0] / batches)  

mouth_pred_fake, mouth_pred_real = tf_cnn_training(training_x, training_y, testing_x, testing_y, batches, epochs,
                    folder_results_fake_test_mouths, folder_results_real_test_mouths, orig_X_test3)



############# Comparing three sets of result ############

'''
print "right after training and testing..."
print "face: ..."
print face_pred_fake, face_pred_real
print "eye: ..."
print eye_pred_fake, eye_pred_real
print "mouth:..."
print mouth_pred_fake, mouth_pred_real
'''

#%%

#print len(face_pred_fake[0])+len(face_pred_real[0])
#%%

pred_fake = []
pred_real = []
for i in range(len(face_pred_fake[0])+len(face_pred_real[0])):
    count = 0
    if (i in face_pred_fake[0]) == True:
        count += 1
    if (i in eye_pred_fake[0]) == True:
        count += 1
    if (i in mouth_pred_fake[0]) == True:
        count += 1
    if count == 3 or count == 2:
        pred_fake.append(i)
    else:
        pred_real.append(i)
print "fake predictions..."
print pred_fake
print "true predictions..."
print pred_real
   
        
# saving original images to the overall test result directory
for i,j in enumerate(pred_fake):                          # save all the predicted "fake" samples into the folder
            plt.imsave(folder_results_fake_test_overall+"/fake_smile_pred_test_"+str(i)
                                                +".jpg",orig_X_test[j,:,:], cmap='gray')

for i,j in enumerate(pred_real):                          # save all the predicted "real" samples into the folder
            plt.imsave(folder_results_real_test_overall+"/real_smile_pred_test_"+str(i)
                                                +".jpg",orig_X_test[j,:,:], cmap='gray') 

# adding corresponding labels from original label dataset to each of the predicted image
labels_pred_fake = []
labels_pred_real = []
for index in pred_fake:
    labels_pred_fake.append(orig_y_test[index])
for index in pred_real:
    labels_pred_real.append(orig_y_test[index])

# computing accuracies + showing all misclassified original images
print "starting with fake labesl...."
total_count = len(pred_fake) + len(pred_real)
correct_count = 0
for i in range(len(pred_fake)):
    if labels_pred_fake[i] == 0.0:
        correct_count += 1
    else:
        print pred_fake[i]
        plt.figure()
        plt.imshow(orig_X_test[pred_fake[i],:,:])
        
prev = correct_count
print "Correct number of predicted fake labels: ..."
print correct_count

print "start with true labels...."
for i in range(len(pred_real)):
    if labels_pred_real[i] == 1.0:
        correct_count += 1
    else:
        print pred_real[i]
        plt.figure()
        plt.imshow(orig_X_test[pred_real[i],:,:])
print "correct number of predicted true labels: ..."
print correct_count-prev
print "Final accuracy is: ", (correct_count*1.0)/total_count