#coding= utf-8  #or gbk 这样才能使用中文

#importing some useful packages
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imutils
import math
import glob as gb

img_no=0
TruePositive=0
leak=0
sum_out=0
sum_train=0
f=open('a.txt','w')

img_path = gb.glob('test/*.ppm')

#计算有多少个正确的
def countsum_train():
    line_train = open('b.txt', 'r').readlines()
    sum_train=len(line_train)
    #print(sum_train)
    return sum_train

#计算检测到了多少个
def countsum_out():
    line_out = open('a.txt', 'r').readlines()
    sum_out=len(line_out)
    #print(sum_out)
    return sum_out

#读取Train每张图片里面的交通标志个数
def traverse_train(name):
    line_train = open('b.txt', 'r').readlines()
    i=0
    train=[]
    for i in range(0,len(line_train)):
        #print(i)
        linea_train=line_train[i].replace("\n","").split(";")
        #print("train",linea_train)
        #print("train",linea_train[0])
        if linea_train[0]==name:
            train.append(linea_train)
    return train

#读取out每张图片里面的交通标志个数
def traverse_out(name):
    line_out = open('a.txt', 'r').readlines()
    i=0
    out=[]
    for i in range(0,len(line_out)):
        #print(i)
        linea_out =line_out[i].replace("test/","").replace("\n","").split(";")
        #print("out",linea_out)
        #print("out",linea_out[0])
        if linea_out[0]==name:
            out.append(linea_out)
    return out

#对比Train和out返回TruePositive的个数
def check(train,out):
    TruePositive=0
    i=0
    ch=0
    cw=0
    n=0
    u=0
    jj=0
    for i in range(0,len(train)):
        #print(i,train[i])
        j=0
        for j in range(0,len(out)):
            #print(j,out[j])
            x1_train=int(train[i][1])
            x1_out=int(out[j][1])
            y1_train=int(train[i][2])
            y1_out=int(out[j][2])
            x2_train=int(train[i][3])
            x2_out=int(out[j][3])
            y2_train=int(train[i][4])
            y2_out=int(out[j][4])
            #print("x1_train",x1_train)
            #print("x2_train",x2_train)
            #print("y1_train",y1_train)
            #print("y2_train",y2_train)
            #print("x1_out",x1_out)
            #print("x2_out",x2_out)
            #print("y1_out",y1_out)
            #print("y2_out",y2_out)
            
            #print("x1_train-x1_out",x1_train-x1_out)
            #print("x2_train-x2_out",x2_train-x2_out)
            #print("x1_train-x2_out",x1_train-x2_out)
            if x1_train-x1_out>0 and x2_train-x2_out>0 and x1_train-x2_out>0:
                ch=0
            if x1_train-x1_out<0 and x2_train-x2_out<0 and x2_train-x1_out<0:
                ch=0
            if x1_train-x1_out<0 and x2_train-x2_out>0:
                ch=x2_out-x1_out
            #print("ch1",ch)
            if x1_train-x1_out<0 and x2_train-x2_out<0 and x2_train-x1_out>0:
                ch=x2_train-x1_out
            #print("ch2",ch)
            if x1_train-x1_out>0 and x2_train-x2_out>0 and x1_train-x2_out<0:
                ch=x2_out-x1_train
            #print("ch3",ch)
            if x1_train-x1_out>0 and x2_train-x2_out<0:
                ch=x2_train-x1_train
            #print("ch4",ch)
            
            
            #print("y1_train-y1_out",y1_train-y1_out)
            #print("y2_train-y2_out",y2_train-y2_out)
            #print("y1_train-y2_out",x1_train-x2_out)
            if y1_train-y1_out>0 and y2_train-y2_out>0 and y1_train-y2_out>0:
                cw=0
            if y1_train-y1_out<0 and y2_train-y2_out<0 and y2_train-y1_out<0:
                cw=0
            if y1_train-y1_out<0 and y2_train-y2_out>0:
                cw=y2_out-y1_out
            #print("cw1",cw)
            if y1_train-y1_out<0 and y2_train-y2_out<0 and y2_train-y1_out>0:
                cw=y2_train-y1_out
            #print("cw2",cw)
            if y1_train-y1_out>0 and y2_train-y2_out>0 and y1_train-y2_out<0:
                cw=y2_out-y1_train
            #print("cw3",cw)
            if y1_train-y1_out>0 and y2_train-y2_out<0:
                cw=y2_train-y1_train
            #print("cw4",cw)
            
            
            if ch!=0 and cw!=0:
                n=ch*cw
                trainArea=(x2_train-x1_train)*(y2_train-y1_train)
                outArea=(x2_out-x1_out)*(y2_out-y1_out)
                u=trainArea+outArea-n
                jj=n/u
                #print("trainArea",trainArea)
                #print("outArea",outArea)
                #print("n",n)
                #print("u",u)
                #print("jj",jj)
                if jj>0.001:
                    TruePositive=TruePositive+1
                    #print("TruePositiveaaaa",TruePositive)
    return (TruePositive)


for path in img_path:
    
    resultall=[]
    #print (path)
    
    img_no=img_no+1;
    
    #reading in an image
    img  = cv2.imread(path)
    img  = cv2.resize(img,None,fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC)
    #高斯模糊
    #img = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow("gaosi",img)
    #cv2.waitKey(0)

    #HSV空间
    #H(色彩/色度) [0，179]， S(饱和度) [0，255]，V(亮度) [0，255]。
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV",hsv)
    #cv2.waitKey(0)

    image1=img.copy()
    image2=img.copy()
    image3=img.copy()

    def color(color,image):
        
        #print (path,color)
        #bule
        lower_blue=np.array([100,43,46])
        upper_blue=np.array([140,255,255])

        #red
        lower_red=np.array([170,40,30])
        upper_red=np.array([180,255,230])
        #lower_red2=np.array([0,43,46])
        #upper_red2=np.array([10,255,255])

        #yellow
        lower_yellow=np.array([18,148,66])
        upper_yellow=np.array([45,255,255])
        
        if color=="blue":
            #根据阈值构建掩模
            #bule
            mask_b=cv2.inRange(hsv,lower_blue,upper_blue)
            #cv2.imshow('mask_b',mask_b)
            #cv2.waitKey(0)
            
            # 对原图像和掩模进行位运算
            res_b=cv2.bitwise_and(image1,image1,mask=mask_b)
            #cv2.imshow('res_b',res_b)
            #cv2.waitKey(0)

            #灰度图
            gray_b = cv2.cvtColor(res_b,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray_b',gray_b)
            #cv2.waitKey(0)
            
            return gray_b
        
        if color=="red":
            #red
            mask_r=cv2.inRange(hsv,lower_red,upper_red)
            #mask_r2=cv2.inRange(hsv,lower_red2,upper_red2)
            #mask_r= cv2.bitwise_or(mask_r,mask_r2)
            #cv2.imshow('mask_r',mask_r)
            #cv2.waitKey(0)
            
            res_r=cv2.bitwise_and(image2,image2,mask=mask_r)
            #cv2.imshow('res_r',res_r)
            #cv2.waitKey(0)
    
            gray_r = cv2.cvtColor(res_r,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray_r',gray_r)
            #cv2.waitKey(0)
            
            return gray_r
            
        if color=="yellow":
            #yellow
            mask_y=cv2.inRange(hsv,lower_yellow,upper_yellow)
            #cv2.imshow('mask_y',mask_y)
            #cv2.waitKey(0)
            
            res_y=cv2.bitwise_and(image3,image3,mask=mask_y)
            #cv2.imshow('res_y',res_y)
            #cv2.waitKey(0)

            gray_y = cv2.cvtColor(res_y,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray_y',gray_y)
            #cv2.waitKey(0)
            
            return gray_y
                
        cv2.destroyAllWindows()

    def processing(gray):
        #二值化
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,thresh1=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        ret,thresh2=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3=cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
        ret,thresh4=cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
        ret,thresh5=cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
        thresh6 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        thresh7 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        
        #titles = ['Original Image','OTSU','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        #images = [gray,thresh, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]
        
        #for i in range(9):
            #plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
            #plt.title(titles[i])
            #plt.xticks([]),plt.yticks([])
        #plt.show()
        #plt.close()
        
        #cv2.imshow('thresh',thresh)
        #cv2.waitKey(0)
        
        cv2.destroyAllWindows()

        #定义结构元素
        #kernel = np.ones((14,5),np.uint8) 线性核
        #cv2.MORPH_RECT,(5, 5) 矩形核
        #构造一个长方形内核。这个内核的宽度大于长度，因此我们可以消除条形码中垂直条之间的缝隙
        #cv2.MORPH_ELLIPSE,(5,5) 椭圆核
        #cv2.MORPH_CROSS,(5,5) 十字核
        
        kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
        #kernel_o1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,5))
        #kernel_o2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
        kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
        #kernel_c1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,8))
        #kernel_c2 = cv2.getStructuringElement(cv2.MORPH_RECT,(8,3))
        
        #开运算 先腐蚀再膨胀
        opened = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_o)
        #opened1 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_o1)
        #opened2 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_o2)
        #opened3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_c)
        #opened4 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_c1)
        #opened5 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_c2)
        
        #titles= ['ELLIPSE33','ELLIPSE56','ELLIPSE65','RECT1212','RECT38','RECT83']
        #images = [opened, opened1, opened2, opened3, opened4, opened5]
        
        #for i in range(6):
            #plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            #plt.title(titles[i])
            #plt.xticks([]),plt.yticks([])
        #plt.show()
        #plt.close()
        
        #cv2.imshow('opened',opened)
        #cv2.waitKey(0)

        #闭运算 先膨胀再腐蚀
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_c)
        #cv2.imshow('closed',closed)
        #cv2.waitKey(0)
        
        return closed

    def shape(color,processed,image):
        
        #print (path,color)
        result=[]
        
        #轮廓检测
        cnts= cv2.findContours(processed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #cnts= cv2.findContours(processed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #print(cnts[0])
        #print(cnts[1])
        #print(cnts[2])

        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # 确保至少有一个轮廓被找到
        if len(cnts)> 0:
            
            #cv2.drawContours(image,cnts[1],-1,(0,0,255),1)
            #cv2.imshow('draw',image)
            #cv2.waitKey(0)
            

            #cv2.drawContours(image,cnts,3,(255,0,0),2)
            #cv2.imshow('ss',image)
            #cv2.waitKey(0)
            
            # 将轮廓按大小降序排序
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                        
            circle_no=0
            triangle_no=0
            rectangle_no=0
            
            draw=img.copy()

            # 对排序后的轮廓循环处理
            for cnt in cnts:
                
                # 获取近似的轮廓
                #epsilon = 0.1*cv2.arcLength(cnt,True)
                #approx = cv2.approxPolyDP(cnt, epsilon, True)

                #cv2.drawContours(image,[cnt],-1,(255,0,0),3)
                #cv2.imshow('everyontours',image)
                #cv2.waitKey(0)
                
                #print("a",cnt)
                #print("b",[cnt])
                
                area=cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)
                x,y,w,h = cv2.boundingRect(cnt)
                # compute the rotated bounding box of the largest contour
                #轮廓转换为矩形
                rect = cv2.minAreaRect(cnt)
                #矩形转换为box
                box = np.int0(cv2.boxPoints(rect))
                #print([box])
                
                #截图存储
                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs)
                x2 = max(Xs)
                y1 = min(Ys)
                y2 = max(Ys)
                hight = y2 - y1
                width = x2 - x1

                #print("area",area)
                #print("perimeter",perimeter)
                #print("hight",hight,"width",width)
                #print("w",w,"h",h)
                
                # draw a bounding box arounded the detected barcode and display the image
                #cv2.drawContours(draw, [box], -1, (255, 0, 0), 2)
                #cv2.imshow("draw", draw)
                #cv2.waitKey(0)
                            
                if perimeter!=0 and float(w*h)!=0 and max(w,h)!=0:
                    #圆形度 c
                    c=(4*math.pi*area)/(perimeter*perimeter)
                    #print("circularity",c)
                    #矩形度 r
                    r=area/float(w*h)
                    #print("Rectangularity",r)
                    #伸长度 e
                    #e=min(width,hight)/max(w,h)
                    e=min(w,h)/max(w,h)
                    #print("elongation",e)
                    
                    out=str(x1)+";"+str(y1)+";"+str(x1+width)+";"+str(y1+hight)
                    
                    if c>=0.85 and r>0.70 and e>0.85:
                        #print(color,"circle")
                        
                        circle_no=circle_no+1
                        #print("circle has",circle_no)
                        
                        result.append([x1,y1,x1+width,y1+hight])
                        #print("current result is",[x1,y1,x1+width,y1+hight])
                        #print("result is",result)
                        
                        f.write(path+";")
                        f.write(out)
                        f.write("\n")
                        
                        c2 = image[y1:y1+hight, x1:x1+width]
                        #cv2.imshow('cut_c',c2)
                        #cv2.waitKey(0)
                        
                        name ='out/'+str(img_no)+color+str(circle_no)+'circle.jpg'
                        #print(name)
                        
                        cv2.imwrite(name,c2)
            
                        # draw a bounding box arounded the detected barcode and display the image
                        cv2.drawContours(draw, [box], -1, (0, 255, 0), 1)
                        cv2.imshow("draw", draw)
                        cv2.waitKey(0)
                    
                    else:
                        #print("circle null")
                        
                        if 0.35<c<0.70 and 0.4<r<0.65 and e>0.8:
                            #print(color,"triangle")
                            
                            triangle_no=triangle_no+1
                            #print("triangle has",triangle_no)
                            
                            result.append([x1,y1,x1+width,y1+hight])
                            #print("current result is",[x1,y1,x1+width,y1+hight])
                            #print("result is",result)
                            
                            f.write(path+";")
                            f.write(out)
                            f.write("\n")
                            
                            c2 = image[y1:y1+hight, x1:x1+width]
                            #cv2.imshow('cut_t',c2)
                            #cv2.waitKey(0)
                            
                            name ='out/'+str(img_no)+color+str(triangle_no)+'triangle.jpg'
                            #print(name)
                            
                            cv2.imwrite(name,c2)
                            
                            # draw a bounding box arounded the detected barcode and display the image
                            cv2.drawContours(draw, [box], -1, (255, 0, 0), 1)
                            cv2.imshow("draw", draw)
                            cv2.waitKey(0)
                        else:
                            #print("triangle null")
                            
                            if 0.6<c<0.85 and r>0.7 and e>0.85:
                                #print(color,"rectangle")
                                
                                rectangle_no=rectangle_no+1
                                #print("rectangle has",rectangle_no)
                                
                                result.append([x1,y1,x1+width,y1+hight])
                                #print("current result is",[x1,y1,x1+width,y1+hight])
                                #print("result is",result)
                                
                                f.write(path+";")
                                f.write(out)
                                f.write("\n")
                       
                                c2 = image[y1:y1+hight, x1:x1+width]
                                #cv2.imshow('cut_r',c2)
                                #cv2.waitKey(0)
                                
                                name ='out/'+str(img_no)+color+str(rectangle_no)+'rectangle.jpg'
                                #print(name)
                                
                                cv2.imwrite(name,c2)
                                
                                # draw a bounding box arounded the detected barcode and display the image
                                cv2.drawContours(draw, [box], -1, (0, 0, 255), 1)
                                cv2.imshow("draw",draw)
                                cv2.waitKey(0)
                            #else:
                                #print("rectangle null")
                                #print(color,"all shape null")
              
        #else :
            #print(color,"can't find counter")
        
        # When everything done, release the capture
        #cap.release()
        cv2.destroyAllWindows()

        #print(img_no, result)
        #return result
             
    gray_b=color("blue",image1)
    gray_r=color("red",image2)
    gray_y=color("yellow",image3)

    processed_b=processing(gray_b)
    processed_r=processing(gray_r)
    processed_y=processing(gray_y)

    reb=shape("blue",processed_b,image1)
    if reb!=[]:
        resultall.append(reb)

    rer=shape("red",processed_r,image2)
    if rer!=[]:
        resultall.append(rer)

    rey=shape("yellow",processed_y,image3)
    if rey!=[]:
        resultall.append(rey)

    #print("result", resultall)

f.close()

sum_out=countsum_out()
#print("sum_out",sum_out)
sum_train=countsum_train()
#print("sum_train",sum_train)

for path in img_path:
    name=path.replace("test/","")
    #print("name",name)
    
    train=traverse_train(name)
    #print("train",train)

    out=traverse_out(name)
    #print("out",out)

    TruePositive=TruePositive+check(train,out)
    #print("TruePositive",TruePositive)


#Recall=TruePositive/sum_train
#print("Recall",Recall)

#Precision=TruePositive/sum_out
#print("Precision",Precision)







