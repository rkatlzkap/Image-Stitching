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

root = Tk()
path = filedialog.askopenfilename(initialdir = "D:/ggggdgg", title = "choose your image", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img3 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "D:/gggfggg", title = "choose your image", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img4 = cv2.imread(path)
root.withdraw()


imga=img4
imgb=img3
imgc=img2
imgd=img1

#cv2.imshow("imga",imga)
#.imshow("imgb",imgb)
#cv2.imshow("imgc",imgc)
#cv2.imshow("imgd",imgd)



#하단부 stitching

sift=cv2.SIFT_create() #sift descriptor 생성

h,w,a=imga.shape #높이와 너비를 구함

kp1,des1=sift.detectAndCompute(imga,None) # imga의 특징점과 디스크립터를 검출
kp2,des2=sift.detectAndCompute(imgb,None) # imgb의 특징점과 디스크립터를 검출

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) # FLANN 매처의 인덱스 파라미터를 설정
search_params=dict(cheks=50) # FLANN 매처의 검색 파라미터를 설정

flann=cv2.FlannBasedMatcher(index_params,search_params) # FLANN 매처 객체를 생성

matches=flann.knnMatch(des1,des2,k=2) #두 image의 매칭점을 찾아서 matches에 할당

good=[]
for m,n in matches:
    if m.distance<0.9*n.distance: #distance가 0.7 이하인 것만 추출
        good.append(m)

if len(good)>MIN_MATCH_COUNT: #4보다 클때 stihcing 수행

    src_pts=np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #첫번째 image의 특징점 좌표
    dst_pts=np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0) # 호모그래피 행렬 M을 구함

    matchesMask=mask.ravel().tolist()


    pts=np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,M)  #호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

    #imgb=cv2.polylines(imgb,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    #두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

else:
    print("Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT))
    matchesMask=None


draw_params=dict(matchColor=(0,255,0),#매칭점 그리기
                 singlePointColor=None,
                 matchesMask=matchesMask,
                 flags=2)

#img3=cv2.drawMatches(imga,kp1,imgb,kp2,good,None,**draw_params)

# plt.imshow(img3,'gray'),plt.show()

width=imgb.shape[1]+imga.shape[1]
height=imgb.shape[0]+imga.shape[0]

dst=cv2.warpPerspective(imga,M,(width,height)) #첫 번째 image을 두 번째 image에 맞게 변환

#cv2.imshow('imga_new',dst),plt.show()


h1,w1,ii=imgb.shape
h2,w2,ii=dst.shape

dst[0:h1,0:w1]=imgb[0:h1,0:w1] # 변환된 두 image을 합성

#cv2.imshow('stiching1',dst),plt.show() #출력

imgn1=dst



# 상단부 stitching

sift=cv2.SIFT_create() #sift descriptor 생성

h,w,a=imgc.shape #높이와 너비를 구함

kp1,des1=sift.detectAndCompute(imgc,None) # imgc의 특징점과 디스크립터를 검출
kp2,des2=sift.detectAndCompute(imgd,None) # imgd의 특징점과 디스크립터를 검출

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) # FLANN 매처의 인덱스 파라미터를 설정
search_params=dict(cheks=50) # FLANN 매처의 검색 파라미터를 설정

flann=cv2.FlannBasedMatcher(index_params,search_params) # FLANN 매처 객체를 생성

matches=flann.knnMatch(des1,des2,k=2) #두 image의 매칭점을 찾아서 matches에 할당

good=[]
for m,n in matches:
    if m.distance<0.9*n.distance: #distance가 0.7 이하인 것만 추출
        good.append(m)


if len(good)>MIN_MATCH_COUNT: #4보다 클때 stihcing 수행

    src_pts=np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #첫번째 image의 특징점 좌표
    dst_pts=np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0) # 호모그래피 행렬 M을 구함

    matchesMask=mask.ravel().tolist()


    pts=np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,M)  #호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

    #imgd=cv2.polylines(imgd,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    #두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

else:
    print("Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT))
    matchesMask=None


draw_params=dict(matchColor=(0,255,0),
                 singlePointColor=None,
                 matchesMask=matchesMask,
                 flags=2)

img3=cv2.drawMatches(imgc,kp1,imgd,kp2,good,None,**draw_params)

#plt.imshow(img3,'gray'),plt.show()

width=imgd.shape[1]+imgc.shape[1]
height=imgd.shape[0]+imgc.shape[0]

dst=cv2.warpPerspective(imgc,M,(width,height)) #첫 번째 image을 두 번째 image에 맞게 변환


#cv2.imshow('imgc_new',dst),plt.show()

h0,w0,ii=imgc.shape    #우선 상단부 이미지 (imgn2)를 구할 때 imgc, imgd, imgc_new의 각 height, width가
h1,w1,ii=imgd.shape    # 중요시 사용될 것이기에 각 parmeter들을 선언하였습니다.
h2,w2,ii=dst.shape

h_end=0

if h0<=h1:
    h_end=h0
else:
    h_end = h1

dst[0:h1,0:w1]=imgd[0:h1,0:w1] # 변환된 두 image을 합성

dst=dst[0:h_end,0:w2]

#cv2.imshow('stiching 2',dst),plt.show() #출력

imgn2=dst


# 상단부 하단부 stitching

sift=cv2.SIFT_create() #sift descriptor 생성

h,w,a=imgn1.shape #높이와 너비를 구함
#print(h)
#print(w)

kp1,des1=sift.detectAndCompute(imgn1,None) # imgn1의 특징점과 디스크립터를 검출
kp2,des2=sift.detectAndCompute(imgn2,None) # imgn2의 특징점과 디스크립터를 검출


matches=flann.knnMatch(des1,des2,k=2) #두 image의 매칭점을 찾아서 matches에 할당

good=[]
for m,n in matches:
    if m.distance<0.9*n.distance: #distance가 0.7 이하인 것만 추출
        good.append(m)

if len(good)>MIN_MATCH_COUNT: #4보다 클때 stihcing 수행

    src_pts=np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #첫번째 image의 특징점 좌표
    dst_pts=np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0) # 호모그래피 행렬 M을 구함

    matchesMask=mask.ravel().tolist()

    pts=np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    #print(pts.shape)  # 4,1,2 # 네개의 꼭지점을 지나는 직사각형을 나타내는데 이것의 배열을 바꿈. 즉 두번째 차원은 1 세번째 차원은 2

    dst=cv2.perspectiveTransform(pts,M)  #호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

    #imgn2=cv2.polylines(imgn2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    #두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

else:
    print("Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT))
    matchesMask=None


draw_params=dict(matchColor=(0,255,0),
                 singlePointColor=None,
                 matchesMask=matchesMask,
                 flags=2)

img77=cv2.drawMatches(imgn1,kp1,imgn2,kp2,good,None,**draw_params)

#plt.imshow(img77,'gray'),plt.show()

width=imgn2.shape[1]+imgn1.shape[1]
height=imgn2.shape[0]+imgn1.shape[0]

dst=cv2.warpPerspective(imgn1,M,(width,height)) #첫 번째 image을 두 번째 image에 맞게 변환
#cv2.imshow('imgn3',dst),plt.show()

#imgn2=imgn2[0:h1,0:w2]
dst[0:imgn2.shape[0],0:imgn2.shape[1]]=imgn2 # 변환된 두 image을 합성

#resize사용해서 크기 낮춤
scale_down = 0.5
scaled_dst = cv2.resize(dst, None, fx=scale_down, fy=scale_down, interpolation= cv2.INTER_LINEAR)

cv2.imshow('stiching result',dst),plt.show() #출력

cv2.waitKey(0)
cv2.destroyAllWindows()
