import cv2
import numpy as np
import time
import os  
#import datetime
#from datetime import datetime
#from PIL import Image
#from io import BytesIO
#from scipy import ndimage
#from pympler.tracker import SummaryTracker
#tracker = SummaryTracker()


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
classesFile = "coco.names"
classes = None


#frame = cv2.imread('1.jpg')
# Give the weight files to the model and load the network using       them.

roi_detection_modelWeights = "/content/best (1).onnx"
roi_detection_model = cv2.dnn.readNet(roi_detection_modelWeights)
roi_detection_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
roi_detection_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#num_rec_modelWeights = "chan_3_32_num.onnx" #ch first
#num_rec_model = cv2.dnn.readNet(num_rec_modelWeights)




def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
def pre_process(input_image, net,w,h):
      # Create a 4D blob from a frame.
      #print(input_image.shape)
      blob = cv2.dnn.blobFromImage(input_image, scalefactor=1/255, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
    #   blob = cv2.dnn.blobFromImage(input_image, 1/255,  (w, h), [0,0,0], 1, crop=False)

      # Sets the input to the network.
      net.setInput(blob)

      # Run the forward pass to get output of the output layers.
      
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      del (blob)
      return outputs

def get_xyxy(input_image, outputs,w,h):
      # Lists to hold respective values while unwrapping.
      class_ids = []
      confidences = []
      boxes = []
      output_boxes=[]
      # Rows.
      rows = outputs[0].shape[1]
      image_height, image_width = input_image.shape[:2]
      # Resizing factor.
      x_factor = image_width / w
      y_factor =  image_height / h
      # Iterate through detections.
      for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
 # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
      indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            # Class label.                      
            #label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])             
            # Draw label.             
            draw_label(input_image, 'x', left, top)
            cv2.imwrite('image.jpg',input_image)
            #turn xywh into xyxy
            boxes[i][2]=left + width
            boxes[i][3]=top + height
            #check if the height is suitable
            output_boxes.append(boxes[i])
            #if height >20:
            #      output_boxes.append(boxes[i])
      #del(input_image,)
      return 1,output_boxes,input_image #boxes (left,top,width,height)

def roi_detection(input_image,roi_detection_model,w,h):
      detections = pre_process(input_image, roi_detection_model,w,h) #detection results
      
      _,bounding_boxes,input_image=get_xyxy(input_image, detections,w,h) # nms and return the valid bounding boxes
      #print( bounding_boxes)
      #date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
      #cv2.imwrite(f"lic_{date}.jpg",image_with_bounding_boxes)
      #cv2.imwrite('xf.jpg',image_with_bounding_boxes)
      return bounding_boxes ,input_image

      
# def number_detection(input_image,ch_detection_model,w,h):
#       #in_image_copy=input_image.copy()
#       detections = pre_process(input_image.copy(), ch_detection_model,w,h) #detection results 
#       image_with_bounding_boxes,bounding_boxes=get_xyxy(input_image, detections,w,h)
#       #date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
#       #im_name=f"ch_{date}.jpg"
#       #print(im_name)
#       #cv2.imwrite(im_name,image_with_bounding_boxes)
#      # cv2.imwrite('x1.jpg',image_with_bounding_boxes)
#       return bounding_boxes






      
def main_func(img,):
      scores='door :'
      img = np.array(img) 
      #send_im_2_tg(img)
      t1=time.time()
      width_height_diff=img.shape[1]-img.shape[0] #padding
      #print(width_height_diff,img.shape)
      if width_height_diff>0:
            img = cv2.copyMakeBorder(img, 0, width_height_diff, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
      if width_height_diff<0:
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(-1*width_height_diff), cv2.BORDER_CONSTANT, (0,0,0))
      cropped_licenses_array,input_image=roi_detection(img.copy(),roi_detection_model,640,640)

      if len(cropped_licenses_array)!=0:
        scores=scores+"True and detection time is  "+str(time.time()-t1)





  

                 
      #print('total time in sec :',time.time()-t1)
      #tracker.print_diff()
      del(img)
      #print(scores)
      #return (scores+'  time_sec : '+str(time.time()-t1))
      return input_image ,scores


import gradio as gr
import cv2
import os 
# im = gr.Image()
def greet(im):
    im=cv2.imread(im)
    
    im,number=main_func(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    

    #print(im)
    
    #im=cv2.imread(im)
    # im=os.path.join("/content/s.jpg")
    
    return im ,number

inputs = gr.Image(type="filepath", label="Input Image")

outputs = [gr.Image(type="filepath", label="Output Image"),gr.Textbox()]
title = "YOLO-v5-Door detection for visually impaired people"

demo_app = gr.Interface(examples=["/content/s.jpg"],
    fn=greet,
    inputs=inputs,
    outputs=outputs,
    title=title,
    cache_examples=True,
)
demo_app.launch()
