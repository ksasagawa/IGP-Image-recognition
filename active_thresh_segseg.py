import cv2
import numpy as np
import time

def segseg(path,dst_size=12, interpol=cv2.INTER_LINEAR):
    start = time.time()
    image = cv2.imread(path)

    pth = path.split('\\')[1][:-4]
    print(pth)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)

    blur = cv2.GaussianBlur(gray, (21,21),0)

    #otsu thresholding - did not quite do enough testing, but I think it's fine for now
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # print(type(cnts))
    ls = []
    l = []
    cuts = []
    Z = 0
    for cnt in cnts:
        i = cv2.contourArea(cnt)
        if i < 100 or i > 10000:
            continue

        # print(Z) 
        # mn = cv2.minAreaRect(cnt)
        # box = np.intp(cv2.boxPoints(mn))
        # cv2.drawContours(image,[box],0,(0,0,255),2)

        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
        # cv2.putText(image, str(Z), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,3,9), 1)

        mask = np.zeros_like(image)
        mask = cv2.drawContours(mask, cnt, -1, (0), -1)
        mask = cv2.bitwise_not(mask)
        # cv2.imshow('image',mask)
        # cv2.waitKey(0)

        out = cv2.bitwise_and(image, mask)
        out = out[y:y+h, x:x+w]

        out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        out = cv2.resize(out, (dst_size,dst_size),interpolation=interpol)

        # cv2.imshow('n',out)
        # cv2.waitKey(0)
        cuts.append(out)

        # cv2.imshow('image',out)
        # cv2.waitKey(0)

        ls.append(i)
        l.append(cnt)


        if not cv2.imwrite("minis\\" + str(pth) + "_" + str(Z) + '.jpg', out):
            print("minis/" + str(pth) + str(Z) + '.jpg')
            raise Exception("no write")

        Z += 1

    print(Z)
    return cuts

    # ls.sort()

    # print(ls)
    # print(len(ls))

    # cv2.drawContours(image, l,-1,[0,255,0],2)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # cv2.imwrite('segseg.png', image)

