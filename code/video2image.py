import sys
import cv2
"""
img_size : tuple
"""

#if __name__ == "__main__":
def video2image(source, target, img_size):
    """
    if len(sys.argv) == 1:
        raise ValueError('Please input at least 1 argument (video source).')
    source = sys.argv[1]
    if len(sys.argv) == 3:
        target = sys.argv[2]
    else:
        target = "video_frames/frame"
    """
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    if (not ret):
        raise ValueError('Failed to read  '+source)
    print 'Frame dimension: ', frame.shape

    count = 0
    #while(ret and cv2.waitKey(1) & 0xFF == ord('q')):
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        resize = cv2.resize(frame, img_size)
        cv2.imwrite(target + '/frame' + str(count) + ".jpg", resize)
        cv2.imshow('frame',resize)
        ret, frame = cap.read()
        count+=1
