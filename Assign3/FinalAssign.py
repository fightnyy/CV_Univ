import numpy as np
import cv2 as cv
import pdb

FLANN_INDEX_LSH    = 6


def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):

    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2



    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k = 2) #2


    matches = []
    drawmatching=[]

    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            drawmatching.append(m[0])


    if len(matches) >= 4:

        keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])


        H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC,4.0)

       
    else:
        H, status = None, None
       


    return matches, H, status , drawmatching


   
def drawMatches(image1, image2, keyPoints1, keyPoints2, matches, status):

    img3 = cv.drawMatches(image1, keyPoints1, image2, keyPoints2, matches, None, flags=10)

    return img3


def main():

    img1 = cv.imread('image1.jpeg') 
    img2 = cv.imread('image2.jpeg')
    
    

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) #이미지 회색으로 바꾸기
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)    #이미지 회색으로 바꾸기

    
 
    detector = cv.ORB_create(
    nfeatures=40000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None) #keyPoints1는 좌표그자체 descriptors 비교하는 keypoints 들이 얼마나 비슷한지 계산하는 것
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
   

    
    
    keyPoints1t = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2t = np.float32([keypoint.pt for keypoint in keyPoints2])
    


    matches, H, status, drawmatching = matchKeypoints(keyPoints1t, keyPoints2t, descriptors1, descriptors2) # 매칭점과 호모그라피를 구한다. 



    img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, drawmatching, status)
    

    


    result = cv.warpPerspective(img1, H,
        (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2


    cv.imshow('matching result', img_matching_result)
    cv.waitKey(0)
    cv.imshow('result', result)
    cv.waitKey(0)
    

    cv.waitKey()

    


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()