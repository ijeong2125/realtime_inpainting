#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sksparse.cholmod import cholesky

def depthcallback(msg):
    global depth_filter
    depth = bridge.imgmsg_to_cv2(msg,msg.encoding)[150:390,320:640]     # [150:390,320:640] 320*240
    # depth = bridge.imgmsg_to_cv2(msg,msg.encoding)[30:510,160:800]     # [30:510,160:800] 640*480
    
    depth_filter_original = cv2.inRange(depth,100,800)

    kernel = []
    kernel.append(np.ones((3,3),np.uint8))
    kernel.append(np.ones((3,3),np.uint8))
    

    Blur = cv2.medianBlur(depth_filter_original,7)
    dilation = cv2.dilate(Blur,kernel[1],iterations=2)
    #dilation = cv2.dilate(dilation,kernel[0],anchor=(2,-1),iterations=1)

    # mask_pub.publish(bridge.cv2_to_imgmsg(depth))
    depth_filter = cv2.bitwise_not(dilation)

def imagecallback(msg):
    global depth_filter
    
    img = bridge.imgmsg_to_cv2(msg,"bgr8")[150:390,320:640]     # [150:390,320:640] 320*240
    # img = bridge.imgmsg_to_cv2(msg,"bgr8")[30:510,160:800]      # [30:510,160:800] 640*480
    img_filtered = cv2.bitwise_and(img, img, mask=depth_filter)
    
    mask_pub.publish(bridge.cv2_to_imgmsg(depth_filter))     
    img_pub.publish(bridge.cv2_to_imgmsg(img_filtered,"bgr8"))
    original_img_pub.publish(bridge.cv2_to_imgmsg(img,"bgr8"))
    
    # img_pub.publish(bridge.cv2_to_imgmsg(img_filtered))
    # original_img_pub.publish(bridge.cv2_to_imgmsg(gray))

if __name__ == '__main__':
    rospy.init_node('depthfilter')
    bridge = CvBridge()
    
    depth_msg = '/kinect2/qhd/image_depth_rect'
    depth_sub = rospy.Subscriber(depth_msg,Image,depthcallback)
    mask_pub = rospy.Publisher('mask',Image,queue_size='10') 
    depth_filter = np.zeros((240,320))

    img_msg = '/kinect2/qhd/image_color_rect'
    img_sub = rospy.Subscriber(img_msg,Image,imagecallback)
    img_pub = rospy.Publisher('img_filtered',Image,queue_size='10')
    original_img_pub = rospy.Publisher('img_original',Image,queue_size='10')

    rate = rospy.Rate(30)
    rospy.spin()
    rate.sleep()
