#coding= utf-8  #or gbk 这样才能使用中文

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imutils
import math
import glob as gb

resultall=[]
img_no=0

img_path = gb.glob('test/123.jpg')
for path in img_path:
    #print (path)

    img_no=img_no+1;
    
    #reading in an image
    img  = cv2.imread(path)
    img  = cv2.resize(img,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_CUBIC)

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
        #lower_blue2=np.array([0,170,170])
        #upper_blue2=np.array([10,240,255])
        
        #red
        lower_red=np.array([156,43,46])
        upper_red=np.array([180,255,255])
        lower_red2=np.array([0,43,46])
        upper_red2=np.array([8,255,255])
        
        #yellow
        lower_yellow=np.array([18,148,60])
        upper_yellow=np.array([45,255,255])
        
        if color=="blue":
            #根据阈值构建掩模
            #bule
            mask_b=cv2.inRange(hsv,lower_blue,upper_blue)
            #mask_b2=cv2.inRange(hsv,lower_blue2,upper_blue2)
            #mask_b= cv2.bitwise_or(mask_b,mask_b2)
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
            mask_r2=cv2.inRange(hsv,lower_red2,upper_red2)
            mask_r= cv2.bitwise_or(mask_r,mask_r2)
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
        #ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret,thresh1=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        #ret,thresh2=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3=cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
        #ret,thresh4=cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
        #ret,thresh5=cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
        #thresh6 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        #thresh7 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        
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
        #opened = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_o)
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
        
        opened = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel_o)
        #cv2.imshow('opened',opened)
        #cv2.waitKey(0)

        #闭运算 先膨胀再腐蚀
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_c)
        #cv2.imshow('closed',closed)
        #cv2.waitKey(0)

        cv2.destroyAllWindows()
        
        return closed

    def shape(color,processed,image):
        
        #print (path,color)
        result=[]
        
        #轮廓检测
        #cnts= cv2.findContours(processed.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnts= cv2.findContours(processed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #print(cnts[0])
        #print(cnts[1])
        #print(cnts[2])

        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # 确保至少有一个轮廓被找到
        if len(cnts)> 0:
            
            draww=img.copy()
            draw=img.copy()
            
            '''
            cv2.drawContours(draww,cnts,-1,(255,0,0),2)
            cv2.imshow('draw',draww)
            cv2.waitKey(0)
            '''
            
            #将轮廓按大小降序排序
            #cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
            circle_no=0
            triangle_no=0
            rectangle_no=0
            
            i=1;
            
            # 对排序后的轮廓循环处理
            for cnt in cnts:

                #print("contour",i)
                #print("aa",cnt.shape)
                
                # 获取近似的轮廓
                #epsilon = 0.1*cv2.arcLength(cnt,True)
                #approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                '''
                a=cv2.drawContours(draww,cnt,-1,(0,0,255),3)
                cv2.imshow('everyontours',draww)
                cv2.waitKey(0)
                '''
            
                area=cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt,True)
                x,y,w,h = cv2.boundingRect(cnt)
                rect = cv2.minAreaRect(cnt)
                box = np.int0(cv2.boxPoints(rect))

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
                '''
                print("perimeter",perimeter)
                print("hight",hight,"width",width)
                print("w",w,"h",h)
                print("box",[box])
                '''
                
                if perimeter!=0 and float(w*h)!=0 and max(w,h)!=0 and area>200:
                    #圆形度 c
                    c=(4*math.pi*area)/(perimeter*perimeter)
                    
                    #矩形度 r
                    r=area/float(w*h)
                    
                    #伸长度 e
                    #e=min(width,hight)/max(w,h)
                    e=min(w,h)/max(w,h)
                    
                    '''
                    print("circularity",c)
                    print("Rectangularity",r)
                    print("elongation",e)
                    '''

                    if c>=0.63 and r>0.66 and e>0.79:
                        #print(color,"circle")

                        '''
                        print("circularity",c)
                        print("Rectangularity",r)
                        print("elongation",e)
                        '''
                        
                        circle_no=circle_no+1
                        #print("circle has",circle_no)
                        
                        result.append([x1,x1+width,y1,y1+hight])
                        #print("result is",result)
                        
                        c2 = image[y1:y1+hight, x1:x1+width]
                        #cv2.imshow('cut_c',c2)
                        #cv2.waitKey(0)
                        
                        name ='out/'+str(img_no)+color+str(circle_no)+'circle.jpg'
                        #print(name)
                        
                        cv2.imwrite(name,c2)
                        
                        abox=[[x1,y1+hight],[x1,y1],[x1+width,y1],[x1+width,y1+hight]]
                        #print("box",[box])
                        abox = np.int0(abox)
                        #print("abox",[abox])
                        # draw a bounding box arounded the detected barcode and display the image
                        #green
                        cv2.drawContours(draw, [abox], -1, (0, 255, 0), 2)
                        cv2.imshow("drawc", draw)
                        cv2.waitKey(0)

                    else:
                        #print("circle null")
                        
                        #if 0.63<c<0.75 and 0.46<r<0.65 and e>0.89:
                        if 0.43<c<0.75 and 0.46<r<0.65 and e>0.88:
                            #print(color,"triangle")

                            '''
                            print("circularity",c)
                            print("Rectangularity",r)
                            print("elongation",e)
                            '''

                            triangle_no=triangle_no+1
                            #print("triangle has",triangle_no)
                            
                            result.append([x1,x1+width,y1,y1+hight])
                            #print("result is",result)
                            
                            c2 = image[y1:y1+hight, x1:x1+width]
                            #cv2.imshow('cut_t',c2)
                            #cv2.waitKey(0)
                            
                            name ='out/'+str(img_no)+str(img_no)+color+str(triangle_no)+'cut_t.jpg'
                            #print(name)
                            
                            cv2.imwrite(name,c2)
                        
                            abox=[[x1,y1+hight],[x1,y1],[x1+width,y1],[x1+width,y1+hight]]
                            #print("box",[box])
                            abox = np.int0(abox)
                            #print("abox",[abox])
                            # draw a bounding box arounded the detected barcode and display the image
                            # bule
                            cv2.drawContours(draw, [abox], -1, (255, 0, 0), 2)
                            cv2.imshow("drawc", draw)
                            cv2.waitKey(0)
                                
                        else:
                            #print("triangle null")
                            
                            if 0.6<c<0.70 and r>0.7 and e>0.85:
                                #print(color,"rectangle")

                                '''
                                print("circularity",c)
                                print("Rectangularity",r)
                                print("elongation",e)
                                '''
                                
                                rectangle_no=rectangle_no+1
                                #print("rectangle has",rectangle_no)
                                
                                result.append([x1,x1+width,y1,y1+hight])
                                #print("result is",result)
                                
                                c2 = image[y1:y1+hight, x1:x1+width]
                                #cv2.imshow('cut_r',c2)
                                #cv2.waitKey(0)
                                
                                name ='out/'+str(img_no)+color+str(rectangle_no)+'cut_r.jpg'
                                #print(name)
                                
                                cv2.imwrite(name,c2)

                                abox=[[x1,y1+hight],[x1,y1],[x1+width,y1],[x1+width,y1+hight]]
                                #print("box",[box])
                                abox = np.int0(abox)
                                #print("abox",[abox])
                                # draw a bounding box arounded the detected barcode and display the image
                                #red
                                cv2.drawContours(draw, [abox], -1, (0, 0, 255), 2)
                                cv2.imshow("drawc", draw)
                                cv2.waitKey(0)

                            #else:
                                #print("rectangle null")
                                #print(color,"all shape null")
                i=i+1
                cv2.destroyAllWindows()
        #else :
            # print(color,"can't find counter")
        #print("resultone", result)
        if(result):
            resultall.append(result)
        # When everything done, release the capture
        #cap.release()
        cv2.destroyAllWindows()
            
    
    gray_b=color("blue",image1)
    gray_r=color("red",image2)
    gray_y=color("yellow",image3)

    processed_b=processing(gray_b)
    processed_r=processing(gray_r)
    processed_y=processing(gray_y)

    shape("blue",processed_b,image1)
    shape("red",processed_r,image2)
    shape("yellow",processed_y,image3)
    
    #print("result", resultall)




