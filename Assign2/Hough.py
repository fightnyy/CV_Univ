"""
실행 방법 :

1. 해당 python 파일이 위치하는 곳에 CardPictues 라는 폴더를 하나 만든다. 
2. 해당 Cardpictures 라는 파일 안에 실험하고자 하는 파일 이름을 Picture'숫자' 로 지정한다.
3. def auto_scan_image로 이동한다. 
4. 첫번째 변수명으로 image가 있다. 이때 PATH 뒤에 있는 숫자를 2에서 지정한 "숫자" 로 바꾸어준다
5. 해당파일을 실행시킨다.
"""

import numpy as np
import cv2
import os 
PATH = 'CardPictures'
PATH = os.path.join(PATH,'Picture')


def order_points(pts):
    pts = pts.tolist()
    rect = np.zeros((4,2), dtype="float32")
    # print("pts.shape : ",pts.shape)
    print("pts. 내용 : ",pts)
    sorted(pts,key= lambda pts: pts[0],reverse=True)
    pts.sort(key=lambda x: x[0:])
    pts = np.array(pts)
    print("good : ",pts[0][1])
    print("change PTS : ",pts)
    if pts[0][1] < pts[1][1]:
        rect[0]=pts[0]
        rect[3]=pts[1]
        print("before : ",rect)
        if pts[2][0] < pts[2][1]:
            rect[1]=pts[3]
            rect[2]=pts[2]
        
        else :
            rect[1] = pts[2]
            rect[2] = pts[3]

    else :
        rect[0] = pts[1]
        rect[3] = pts[0] 

        if pts[2][0] < pts[2][1]:
            rect[1]=pts[3]
            rect[2]=pts[2]
        
        else :
            rect[1] = pts[2]
            rect[2] = pts[3]
    
    # s = pts.sum(axis =1)
    # print("what is s : ",s)
    # rect[0]=pts[np.argmin(s)]
    # rect[2]=pts[np.argmax(s)]

    # diff= np.diff(pts, axis =1 )
    # # rect[1]=pts[np.argmin(diff)]
    # # rect[3]=pts[np.argmax(diff)]
    
    # rect[0]=pts[0]
    # rect[2]=pts[3]

    return rect


def auto_scan_image():
    image=cv2.imread(PATH+"4.jpg")
    orig = image.copy()
    f = float(image.shape[0])
    i = image.shape[0]
    r = f /image.shape[0]
    dim = (int(image.shape[1]*r),i)
    image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edged = cv2.Canny(gray,40, 100)
 
    

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]
    

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx)==4:
            screenCnt = approx
            break
    
    cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
   

    cv2.imshow("Lines",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rect = order_points(screenCnt.reshape(4,2)/r)

    (topLeft, topRight, bottomRight, bottomLeft) =rect


    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])


    maxWidth = max([w1,w2])
    maxHeight = max([h1,h2])

    dst = np.float32([[0, 0], [maxWidth-1 ,0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]])

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    cv2.imshow("Warped",warped)
    cv2.imwrite("Warped.png",warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)



if __name__ == "__main__":
    auto_scan_image()