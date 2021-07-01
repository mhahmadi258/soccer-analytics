import numpy as np
import cv2

N = 1050
M = 680
FILTER_SIZE = 7

output_size = (N,M)

fg = cv2.getGaussianKernel(FILTER_SIZE, sigma= -1)
fg = fg.dot(fg.T)

closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,23))
opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


# Create BackgroundSubtractor object
backSub = cv2.bgsegm.createBackgroundSubtractorMOG()


# Find perspective transformation
src_point = np.array([(872,778),(1227,86),(141,167),(1135,115)], dtype=np.float32)
dest_points = np.array([(525,680),(1050,0),(164,139),(886,139)], dtype=np.float32)
H = cv2.getPerspectiveTransform(src_point, dest_points)

# Create a VideoCapture object
cap = cv2.VideoCapture('src/sample1.mp4')

if not cap.isOpened():
    cap.open()

while True:
    # Capture frame-by-frame
    ret, I = cap.read()

    if ret == False:    # End of video
        break


    J = cv2.warpPerspective(I, H, output_size)  # Apply transformation


    fgmask = backSub.apply(I)   # Apply background subtraction

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, opening_kernel)   # Apply opening
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, closing_kernel)  # Apply closing
    
    fgmask = cv2.filter2D(fgmask,-1,fg) # Filtering with Gaussian kernel

    nc, CC, stats, centorids = cv2.connectedComponentsWithStats(fgmask) # Find connected components
    
    stats[:,1] = stats[:,1] + stats[:,3]
    stats[:,0] = stats[:,0] + stats[:,2] // 2
    pts = stats[:,[0,1]]
    pts = pts.reshape(-1,1,2)
    pts = pts.astype('float64')

    print(pts.dtype)

    pts = cv2.perspectiveTransform(pts,H).reshape(-1,2)

    pts = pts[1:]

    for pt in pts:
        cv2.circle(J,(int(pt[0]),int(pt[1])),5,(0,0,255))
    
    cv2.imshow('j',J)
    # cv2.imshow('Frame',K)
    cv2.imshow('FG Mask',fgmask)

    key = cv2.waitKey(33)   # ~ 30 frames per second
    if key & 0xFF == ord('q'):  # exit when "q" is pressed
        break

cap.release()


cv2.destroyAllWindows()

