import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image

MIN_MATCH_COUNT=4 #최소한으로 필요한 매칭점의 수를 4개로 정의


root = Tk()
path = filedialog.askopenfilename(initialdir = "D:/DATA_", title = "choose your image", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img1 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "D:/gggggg", title = "choose your image", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img2 = cv2.imread(path)
root.withdraw()

#cv2.imshow("img1",img1)
#cv2.imshow("img2",img2)



sift=cv2.SIFT_create() #sift descriptor 생성

kp1,des1=sift.detectAndCompute(img1,None) # img1의 특징점과 디스크립터를 검출
kp2,des2=sift.detectAndCompute(img2,None) # img2의 특징점과 디스크립터를 검출

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) # FLANN 매처의 인덱스 파라미터를 설정
search_params=dict(cheks=500) # FLANN 매처의 검색 파라미터를 설정

flann=cv2.FlannBasedMatcher(index_params,search_params) # FLANN 매처 객체를 생성

matches=flann.knnMatch(des1,des2,k=2) #두 image의 매칭점을 찾아서 matches에 할당

good=[]
for m,n in matches:
    if m.distance<0.7*n.distance: #distance가 0.7 이하인 것만 추출
        good.append(m)

if len(good)>MIN_MATCH_COUNT: #4보다 클때 stihcing 수행

    src_pts=np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #첫번째 image의 특징점 좌표
    dst_pts=np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0) # 호모그래피 행렬 M을 구함

    matchesMask=mask.ravel().tolist()

    h,w,a=img1.shape #높이와 너비를 구함
    pts=np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,M)  #호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

    #img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    #두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

else:
    print("Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT))
    matchesMask=None



#print(M)
answer1=''
answer2=''

x=M[0,2]
y=M[1,2]

if abs(x)>50:
    if x>0:
        answer1+='우'
        answer2+='좌'
    else:
        answer1+='좌'
        answer2 += '우'
if abs(y) > 50:
    if y>0:
        answer1+='하'
        answer2 += '상'
    else:
        answer1+='상'
        answer2 += '하'

print('img1 = ',answer1)
print('img2 = ',answer2)





cv2.waitKey(0)
cv2.destroyAllWindows()