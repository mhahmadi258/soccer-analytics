import numpy as np
import cv2
from keras.models import load_model
import time

N = 1050
M = 700

output_size = (N,M)

# this method prepare image size for model input
def prepare_image(imgs):
    imgs =np.array([cv2.resize(img, dsize=(36,58)) for img in imgs])
    return imgs

# Create I shape closing kernel
closing_kernel = np.zeros((25,15),dtype=np.uint8)
closing_kernel[:6,:] = closing_kernel[-6:,:] = closing_kernel[:,3:12] = 1

# Create BackgroundSubtractor object
backSub = cv2.bgsegm.createBackgroundSubtractorMOG()


# Find perspective transformation
src_point = np.array([(872,778),(1227,86),(141,167),(1135,115)], dtype=np.float32)
dest_points = np.array([(525,700),(1050,0),(164,143),(886,143)], dtype=np.float32)
H = cv2.getPerspectiveTransform(src_point, dest_points)

# read classification model from file 
model = load_model('model/model')

# Create a VideoCapture object
cap = cv2.VideoCapture('src/sample1.mp4')

if not cap.isOpened():
    cap.open()

while True:
    # Capture frame-by-frame
    ret, I = cap.read()

    if ret == False:    # End of video
        break


    out_image = cv2.imread('src/2D_field.png')

    J = cv2.warpPerspective(I, H, output_size)  # Apply transformation


    fgmask = backSub.apply(I)   # Apply background subtraction

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, closing_kernel)  # Apply closing
    
    nc, CC, stats, centorids = cv2.connectedComponentsWithStats(fgmask) # Find connected components
    
    boxes = stats[1:,:4].copy()
    boxes[:,2] = boxes[:,0] + boxes[:,2]    # Right of image
    boxes[:,3] = boxes[:,1] + boxes[:,3]    # Bottom of image

    pts = stats.copy()
    pts[:,1] = stats[:,1] + stats[:,3]    # Bottom of components
    pts[:,0] = stats[:,0] + stats[:,2] // 2   # Middle of components in x dirction
    pts = pts[0:,[0,1]]

    hws = stats[1:,2:4]     # Height and Width of componenets
    pts = pts.reshape(-1,1,2)
    pts = pts.astype('float64')


    pts = cv2.perspectiveTransform(pts,H).reshape(-1,2)     # Apply transformation to the points

    pts = pts[1:]   # remove background component point

    # compute croped images
    images = list()
    for box,hw in zip(boxes,hws):
        croped_image = I[box[1]:box[3],box[0]:box[2],:]
        images.append(croped_image)

    # compute labales
    images = prepare_image(images)
    labels = np.array([])
    if len(images) > 0:
        softmaxs = model.predict(images)
        labels = np.argmax(softmaxs,axis=1)

    # add circle to output image
    for pt, label, hw, box in zip(pts,labels,hws,boxes):
        if abs(hw[0] - hw[1]) <= 10 :    # remove the  ball
            continue
        if label == 0:
            cv2.circle(out_image,(int(pt[0]),int(pt[1])),15,(255,0,0),-1)   # Blue
        elif label == 1:
            cv2.circle(out_image,(int(pt[0]),int(pt[1])),15,(255,255,255),-1)   # White
        elif label == 2:
            cv2.circle(out_image,(int(pt[0]),int(pt[1])),15,(0,255,255),-1) # Yellow
        else :
            cv2.circle(out_image,(int(pt[0]),int(pt[1])),15,(0,0,255),-1)   # Red
        cv2.rectangle(I, (box[:2]), (box[2:]), (0, 255, 0), 2)
        cv2.circle(J,(int(pt[0]),int(pt[1])),5,(0,0,255))

    

# Display images 
    cv2.imshow('I',I)
    cv2.imshow('j',J)
    cv2.imshow('out',out_image)
    cv2.imshow('FG Mask',fgmask)
    

    key = cv2.waitKey(33)   # ~ 30 frames per second
    if key & 0xFF == ord('q'):  # exit when "q" is pressed
        break

cap.release()


cv2.destroyAllWindows()

