import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image

## 일단 img4개를 입력받습니다

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


## 1,2 판단

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

#print('img1 = ',answer1)
#print('img2 = ',answer2)


## 3,4 판단


kp1,des1=sift.detectAndCompute(img3,None) # img3의 특징점과 디스크립터를 검출
kp2,des2=sift.detectAndCompute(img4,None) # img4의 특징점과 디스크립터를 검출

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

    h,w,a=img3.shape #높이와 너비를 구함
    pts=np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,M)  #호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

    #img4=cv2.polylines(img4,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    #두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

else:
    print("Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT))
    matchesMask=None


#print(M)
answer3=''
answer4=''

x=M[0,2]
y=M[1,2]

if abs(x)>50:
    if x>0:
        answer3+='우'
        answer4+='좌'
    else:
        answer3+='좌'
        answer4 += '우'
if abs(y) > 50:
    if y>0:
        answer3+='하'
        answer4 += '상'
    else:
        answer3+='상'
        answer4 += '하'

#print('img3 = ',answer1)
#print('img4 = ',answer2)



##  case1  1이랑2가 좌우인경우 - 3이랑 4도 좌우


if answer1=='좌' or answer1=='우':


    # 순서 1번  img1랑 img3을 비교해서 1-2 랑 3-4 중에 어떤 세트가 더 위인지 판단

    sift = cv2.SIFT_create()  # sift descriptor 생성

    kp1, des1 = sift.detectAndCompute(img1, None)  # img1의 특징점과 디스크립터를 검출
    kp2, des2 = sift.detectAndCompute(img3, None)  # img3의 특징점과 디스크립터를 검출

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # FLANN 매처의 인덱스 파라미터를 설정
    search_params = dict(cheks=500)  # FLANN 매처의 검색 파라미터를 설정

    flann = cv2.FlannBasedMatcher(index_params, search_params)  # FLANN 매처 객체를 생성

    matches = flann.knnMatch(des1, des2, k=2)  # 두 image의 매칭점을 찾아서 matches에 할당

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # distance가 0.7 이하인 것만 추출
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:  # 4보다 클때 stihcing 수행

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # 첫번째 image의 특징점 좌표
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 호모그래피 행렬 M을 구함

        matchesMask = mask.ravel().tolist()

        h, w, a = img1.shape  # 높이와 너비를 구함
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # 호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

        # img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
        # 두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # print(M)
    answer12 = ''
    answer34 = ''

    x = M[0, 2]
    y = M[1, 2]

    if abs(x) > 50:
        if x > 0:
            answer12 += '우'
            answer34 += '좌'
        else:
            answer12+= '좌'
            answer34+= '우'
    if abs(y) > 50:
        if y > 0:
            answer12+= '하'
            answer34+= '상'
        else:
            answer12+= '상'
            answer34+= '하'

    print('img12 = ', answer12)
    print('img34 = ', answer34)



    # 순서 2번  좌우측 맞춰주기

    if answer1 == '좌':  # 순서 바꿔주기 !! (만약에 img1이 좌측이면 바꿔줌 img1이 우측, img2가 좌측되게끔)
        temp = img1
        img1 = img2
        img2 = temp


    if answer3 == '좌':  # 순서 바꿔주기 !! (만약에 img3이 좌측이면 바꿔줌 img3이 우측, img4가 좌측되게끔)
        temp = img3
        img3 = img4
        img4 = temp



    # 순서 3번 위아래정보 (img12)를 고려하여 abcd에 할당

    if answer12 == '상':  # 순서 바꿔주기 !! (만약에 img12가 위면 바꿔줌 img12가 아래로 오게끔)
        temp = img1
        img1 = img3
        img3 = temp

        temp = img2
        img2 = img4
        img4 = temp

    imga = img1
    imgb = img2
    imgc = img3
    imgd = img4




##  case2  1이랑2가 상하인경우 - 3이랑 4도 상하


if answer1 == '상' or answer1 == '하':


    # 순서 1번  img1랑 img3을 비교해서 1-2 랑 3-4 중에 어떤 세트가 우측인지 판단

    sift = cv2.SIFT_create()  # sift descriptor 생성

    kp1, des1 = sift.detectAndCompute(img1, None)  # img1의 특징점과 디스크립터를 검출
    kp2, des2 = sift.detectAndCompute(img3, None)  # img3의 특징점과 디스크립터를 검출

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # FLANN 매처의 인덱스 파라미터를 설정
    search_params = dict(cheks=500)  # FLANN 매처의 검색 파라미터를 설정

    flann = cv2.FlannBasedMatcher(index_params, search_params)  # FLANN 매처 객체를 생성

    matches = flann.knnMatch(des1, des2, k=2)  # 두 image의 매칭점을 찾아서 matches에 할당

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # distance가 0.7 이하인 것만 추출
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:  # 4보다 클때 stihcing 수행

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # 첫번째 image의 특징점 좌표
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 호모그래피 행렬 M을 구함

        matchesMask = mask.ravel().tolist()

        h, w, a = img1.shape  # 높이와 너비를 구함
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # 호모그래피 행렬 M을 이용하여 pts의 좌표를 두 번째 image에 맞게 변환

        # img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
        # 두 번째 image에서 변환된 pts의 좌표를 다각형으로 그림

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # print(M)
    answer12 = ''
    answer34 = ''

    x = M[0, 2]
    y = M[1, 2]

    if abs(x) > 50:
        if x > 0:
            answer12 += '우'
            answer34 += '좌'
        else:
            answer12+= '좌'
            answer34+= '우'
    if abs(y) > 50:
        if y > 0:
            answer12+= '하'
            answer34+= '상'
        else:
            answer12+= '상'
            answer34+= '하'

    print('img12 = ', answer12)
    print('img34 = ', answer34)



    # 순서 2번  위아래 맞춰주기

    if answer1 == '상':  # 순서 바꿔주기 !! (만약에 img1이 위측이면 바꿔줌 img1이 아래, img2가 위측이 되게끔)
        temp = img1
        img1 = img2
        img2 = temp


    if answer3 == '상':  # 순서 바꿔주기 !! (만약에 img3이 위측이면 바꿔줌 img3이 아래, img4가 위측이 되게끔)
        temp = img3
        img3 = img4
        img4 = temp



    # 순서 3번 좌우측 정보 (img12)를 고려하여 abcd에 할당

    if answer12 == '좌':  # 순서 바꿔주기 !! (만약에 img12가 좌측이면 바꿔줌 img12가 우측으로 오게끔)
        temp = img1
        img1 = img3
        img3 = temp

        temp = img2
        img2 = img4
        img4 = temp

    imga = img1
    imgb = img3
    imgc = img2
    imgd = img4




## case 3
if answer1=='우하' or answer1=='좌하' or answer1=='우상' or answer1=='좌상':

    aa = [[img1, answer1], [img2, answer2], [img3, answer3], [img4, answer4]]

    answers = ['우하', '좌하', '우상', '좌상']

    for i in range(4):
        answer = answers[i]
        for element in aa:
            if element[1] == answer:
                if i == 0:
                    imga = element[0]
                elif i == 1:
                    imgb = element[0]
                elif i == 2:
                    imgc = element[0]
                elif i == 3:
                    imgd = element[0]
                break



## a,b,c,d 다 구하고 나서 이후론 그냥 level_A1b 넣으면됨
# level_A1b 그대로 가져옴 (물론 앞부분은 빼고)  여기서부터 끊은거 !!!!!!

#cv2.imshow("imga",imga)
#cv2.imshow("imgb",imgb)
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