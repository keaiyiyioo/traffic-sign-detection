import tkinter.filedialog as askopenfilename
from tkinter import *
import os
from PIL import Image, ImageTk

import tensorflow as tf
from tensorflow.contrib.layers import flatten

import skimage.transform
import skimage
import time
from datetime import timedelta
import numpy as np



import tkinter.messagebox #这个是消息框，对话框的关键

import matplotlib
root = Tk()

#total detected pic
global nextclick #next button 点击一下加一

nextclick=1

#训练结果对应的文本框
coarse_result = StringVar()
coarse_result .set('')

coarse_info = StringVar()
coarse_info.set('')

fine_result = StringVar()
fine_result .set('')

fine_info = StringVar()
fine_info.set('')

detected_img = StringVar()
detected_img.set('')


showing_img = StringVar()
showing_img.set('')


'''
#实际结果对应的文本框
real_result = StringVar()
real_result.set('')
#检查结果对应的文本框
check_result = StringVar()
check_result.set('')
'''




#匹配序号与含义
def fine_index(x):
    if x == [0]:
        stu = "Speed limit (20km/h)"
        return stu
    if x == [1]:
        stu = "Speed limit (30km/h)"
        return stu
    if x == [2]:
        stu = "Speed limit (50km/h)"
        return stu
    if x == [3]:
        stu = "Speed limit (60km/h)"
        return stu
    if x == [4]:
        stu = "Speed limit (70km/h)"
        return stu
    if x == [5]:
        stu = "Speed limit (80km/h)"
        return stu
    if x == [6]:
        stu = "End of speed limit (80km/h)"
        return stu
    if x == [7]:
        stu = "Speed limit (100km/h)"
        return stu
    if x == [8]:
        stu = "Speed limit (120km/h)"
        return stu
    if x == [9]:
        stu = "No passing"
        return stu
    if x == [10]:
        stu = "No passing for vehicles over 3.5 metric tons"
        return stu
    if x == [11]:
        stu = "Right-of-way at the next intersection"
        return stu
    if x == [12]:
        stu = "Priority road"
        return stu
    if x == [13]:
        stu = "Yield"
        return stu
    if x == [14]:
        stu = "Stop"
        return stu
    if x == [15]:
        stu = "No vehicles"
        return stu
    if x == [16]:
        stu = "Vehicles over 3.5 metric tons prohibited"
        return stu
    if x == [17]:
        stu = "No entry"
        return stu
    if x == [18]:
        stu = "General caution"
        return stu
    if x == [19]:
        stu = "Dangerous curve to the left"
        return stu
    if x == [20]:
        stu = "Dangerous curve to the right"
        return stu
    if x == [21]:
        stu = "Double curve"
        return stu
    if x == [22]:
        stu = "Bumpy road"
        return stu
    if x == [23]:
        stu = "Slippery road"
        return stu
    if x == [24]:
        stu = "Road narrows on the right"
        return stu
    if x == [25]:
        stu = "Road work"
        return stu
    if x == [26]:
        stu = "Traffic signals"
        return stu
    if x == [27]:
        stu = "Pedestrians"
        return stu
    if x == [28]:
        stu = "Children crossing"
        return stu
    if x == [29]:
        stu = "Bicycles crossing"
        return stu
    if x == [30]:
        stu = "Beware of ice/snow"
        return stu
    if x == [31]:
        stu = "Wild animals crossing"
        return stu
    if x == [32]:
        stu = "End of all speed and passing limits"
        return stu
    if x == [33]:
        stu = "Turn right ahead"
        return stu
    if x == [34]:
        stu = "Turn left ahead"
        return stu
    if x == [35]:
        stu = "Ahead only"
        return stu
    if x == [36]:
        stu = "Go straight or right"
        return stu
    if x == [37]:
        stu = "Go straight or left"
        return stu
    if x == [38]:
        stu = "Keep right"
        return stu
    if x == [39]:
        stu = "Keep left"
        return stu
    if x == [40]:
        stu = "Roundabout mandatory"
        return stu
    if x == [41]:
        stu = "End of no passing"
        return stu
    if x == [42]:
        stu = "End of no passing by vehicles over 3.5 metric tons"
        return stu

def coarse_index(x):
    if x == [1]:
        stu = "Speed limitation"
        return stu
    if x == [2]:
        stu = "End limitation"
        return stu
    if x == [3]:
        stu = "Other limitation"
        return stu
    if x == [4]:
        stu = "Direction signs"
        return stu
    if x == [5]:
        stu = "Warning signs"
        return stu
    if x == [6]:
        stu = "Other signs"
        return stu


def coarse_model_index(x):
    if x == [1]:
        stu = "00001"
        return stu
    if x == [2]:
        stu = "00002"
        return stu
    if x == [3]:
        stu = "00003"
        return stu
    if x == [4]:
        stu = "00004"
        return stu
    if x == [5]:
        stu = "00005"
        return stu
    if x == [6]:
        stu = "00006"
        return stu

def fine_model_index(x):
    if x == [0]:
        stu = "00000"
        return stu
    if x == [1]:
        stu = "00001"
        return stu
    if x == [2]:
        stu = "00002"
        return stu
    if x == [3]:
        stu = "00003"
        return stu
    if x == [4]:
        stu = "00004"
        return stu
    if x == [5]:
        stu = "00005"
        return stu
    if x == [6]:
        stu = "00006"
        return stu
    if x == [7]:
        stu = "00007"
        return stu
    if x == [8]:
        stu = "00008"
        return stu
    if x == [9]:
        stu = "00009"
        return stu
    if x == [10]:
        stu = "00010"
        return stu
    if x == [11]:
        stu = "00011"
        return stu
    if x == [12]:
        stu = "00012"
        return stu
    if x == [13]:
        stu = "00013"
        return stu
    if x == [14]:
        stu = "00014"
        return stu
    if x == [15]:
        stu = "00015"
        return stu
    if x == [16]:
        stu = "00016"
        return stu
    if x == [17]:
        stu = "00017"
        return stu
    if x == [18]:
        stu = "00018"
        return stu
    if x == [19]:
        stu = "00019"
        return stu
    if x == [20]:
        stu = "00020"
        return stu
    if x == [21]:
        stu = "00021"
        return stu
    if x == [22]:
        stu = "00022"
        return stu
    if x == [23]:
        stu = "00023"
        return stu
    if x == [24]:
        stu = "00024"
        return stu
    if x == [25]:
        stu = "00025"
        return stu
    if x == [26]:
        stu = "00026"
        return stu
    if x == [27]:
        stu = "00027"
        return stu
    if x == [28]:
        stu = "00028"
        return stu
    if x == [29]:
        stu = "00029"
        return stu
    if x == [30]:
        stu = "00030"
        return stu
    if x == [31]:
        stu = "00031"
        return stu
    if x == [32]:
        stu = "00032"
        return stu
    if x == [33]:
        stu = "00033"
        return stu
    if x == [34]:
        stu = "00034"
        return stu
    if x == [35]:
        stu = "00035"
        return stu
    if x == [36]:
        stu = "00036"
        return stu
    if x == [37]:
        stu = "00037"
        return stu
    if x == [38]:
        stu = "00038"
        return stu
    if x == [39]:
        stu = "00039"
        return stu
    if x == [40]:
        stu = "00040"
        return stu
    if x == [41]:
        stu = "00041"
        return stu
    if x == [42]:
        stu = "00042"
        return stu
    if x == [43]:
        stu = "00043"
        return stu

def resize(w, h, w_box, h_box, pil_image):
    '''''
        resize a pil_image object so it will fit into
        a box of size w_box times h_box, but retain aspect ratio
        对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
        '''
    f1 = 1.0*w_box/w # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    #print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


#选择图片
def callback():
   
    #调用filedialog模块的askdirectory()函数去选择图片
    global originalimg
    #使用全局变量filepath保存图片路径
    originalimg = filedialog.askopenfilename()
    print (originalimg)
    #打开图片（根据filepath）
    create_image_label()

#定义一个frame显示ui内容，button，图片，文本框
image_frame = Frame(root)

#将选择的图片显示出来
image_file = im = image_label = None
def create_image_label():
    global image_file, im, image_label
    image_file = Image.open(originalimg)
    
    w_box = 100
    h_box = 100
    w,h =image_file.size
    
    image_file_resize=resize(w, h, w_box, h_box, image_file)
    
    im = ImageTk.PhotoImage(image_file_resize)
    image_label = Label(image_frame,image = im)
    image_label.grid(row = 4, column = 0, sticky = NW, pady = 8, padx = 20)


image_file2 = im2 = image_label2 = None

def show_image_detected():
    global nextclick
    path = "/Users/apple/Desktop/final-fyp/out"
    print("path is ", path)
    file_names=[]
    file_num=0
    
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)):
            file_names.append(os.path.join(path,i))
            file_num=file_num+1
            print("file_num is ", file_num)
            print("file_names is ", file_names)


    detected_img.set(file_num)
    showing_img.set(nextclick)


    global image_file2, im2, image_label2
    image_file2 = Image.open(file_names[nextclick])

    global filepath
    #使用全局变量filepath保存图片路径
    filepath = file_names[nextclick]
    print (filepath)

    #计算path这个文件夹中的图片数量
    im2 = ImageTk.PhotoImage(image_file2)
    image_label2 = Label(image_frame,image = im2)
    image_label2.grid(row = 4, column = 1, sticky = NW, pady = 8, padx = 20)

    print("nextclick",nextclick)
    if(nextclick<len(file_names)-1):
        nextclick=nextclick+1
    else:
        nextclick=0


def show_next_img():
    show_image_detected()

'''
粗分类,与细分类的区别主要在于
    1.读取的模型不同，（粗分类只有一个模型，细分类有六个模型）
    2.n_class 不同(粗分类为6， 细分类为43)
'''
def Testpic():
        Coarse_classification()


def Coarse_classification():
    
    n_classes = 8
    
    graph = tf.Graph()
    # Create model in the graph.
    with graph.as_default():
        
        #images_ph的占位符形式大概是[None, 32, 32, 3]，这个分别表示[批次，高，宽，通道]
        #批次是None表示批次是灵活的，这就意味着我们可以在不改变代码的情况下修改批次。
        #请注意参数的顺序，因为在像NCHW这样的模型中，参数的顺序是不同的。
        #接下来，我定义了全连接层。与往常不同，我没有实现y = xW + b等式，我使用了一个简单的非线性函数来实现激活函数的功能。我期望输入是个1维的数组，所以首先我将图片平整化。
        #ReLU 作为激活函数。
        #所有负数的函数值都是0，这样对于分类任务和训练速度来说都会强于sigmoid和tanh函数
        
        # 设置占位符用来放置图片和标签，占位符石tensorflow从主程序中接受输入的方式
        # 在graph.as_default() 中创建的占位符（和其他所有操作）， 这样的好处是他们成为了创建图的一部分，而不是在全局图中
        #参数 images_ph 的维度是 [None, 32, 32, 3]，这四个参数分别表示 [批量大小，高度，宽度，通道] （通常缩写为 NHWC）。批处理大小用 None 表示，意味着批处理大小是灵活的，也就是说，我们可以向模型中导入任意批量大小的数据，而不用去修改代码。注意你输入数据的顺序，因为在一些模型和框架下面可以使用不同的排序，比如 NCHW。
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_ph = tf.placeholder(tf.int32, [None])
        
        #接下来，我定义一个全连接层，而不是实现原始方程 y = xW + b。在这一行中，我使用一个方便的函数，并且使用激活函数。模型的输入时一个一维向量，所以我先要压平图片。
        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)
        
        #全连接层的输出是一个长度是62的对数矢量（从技术上分析，它的输出维度应该是 [None, 62]，因为我们是一批一批处理的）。
        #输出的数据可能看起来是这样的：[0.3, 0, 0, 1.2, 2.1, 0.01, 0.4, ... ..., 0, 0]。值越高，图片越可能表示该标签。
        # Fully connected layer.
        # Generates logits of size [None, 62]
        # logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
        
        logits = LeNet(images_ph, n_classes)
        
        #在这个项目中，我们只需要知道最大值所对应的索引就行了，因为这个索引代表着图片的分类标签，这个求解最大操作可以如下表示：
        #argmax 函数的输出结果将是一个整数，范围是 [0, 61]。
        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        predicted_labels = tf.argmax(logits, 1)
        
        #我们需要将标签和神经网络的输出转换成概率向量。TensorFlow中有一个 sparse_softmax_cross_entropy_with_logits 函数可以实现这个操作。这个函数将标签和神经网络的输出作为输入参数，并且做三件事：第一，将标签的维度转换为 [None, 62]（这是一个0-1向量）；第二，利用softmax函数将标签数据和神经网络输出结果转换成概率值；第三，计算两者之间的交叉熵。这个函数将会返回一个维度是 [None] 的向量（向量长度是批处理大小），然后我们通过 reduce_mean 函数来获得一个值，表示最终的损失值。
        # Define the loss function.
        # Cross-entropy is a good choice for classification.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
        
        #下一个需要处理的就是选择一个合适的优化算法。我一般都是使用 ADAM 优化算法，因为它的收敛速度比一般的梯度下降法更快。如果你想知道不同优化器之间的比较结果，
        # Create training op.
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        
        #图中的最后一个节点是初始化所有的操作，它简单的将所有变量的值设置为零（或随机值）。
        # And, finally, an initialization op to execute before training.
        # TODO: rename to tf.global_variables_initializer() on TF 0.12.
        init = tf.initialize_all_variables()
        
        #模型保存加载工具
        saver = tf.train.Saver()
    print()
    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", predicted_labels)
    print()
    
    #请注意，上面的代码还没有执行任何操作。它只是构建图，并且描述了输入。在上面我们定义的变量，比如，init，loss和predicted_labels，它们都不包含具体的数值。它们是我们接下来要执行的操作的引用。


    #这是我们迭代训练模型的地方。在我们开始训练之前，我们需要先创建一个会话（Session）对象。
    #开始训练


    #初始化
    session = tf.Session(graph=graph)
    if os.path.exists('tmp/checkpoint'): #判断模型是否存在
        saver.restore(session, 'tmp/model.ckpt') #存在就从模型中恢复变量
        print()
        print("model has been loaded ! ")
    else:
        print()
        print("no model")

    #评估
    # Load the test dataset.
    #pil 读取图片，并将图片转位numpy数组
    im = Image.open(filepath)
    im_array = np.array(im)

    #测试的输入，图片（numpy数组形式），label
    image=im_array
    label=int(6)

    # Transform the images, just like we did with the training set.
    test_images32 = [skimage.transform.resize(image, (32, 32))]

    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]
    #显示到训练结果的文本框中
    coarse_predict_result = predicted
    coarse_result.set(predicted)
    coarse_info.set(coarse_index(predicted))

    print("predicted label is ", predicted)
    # Close the session. This will destroy the trained model.
    print()
    print("predicted info is ", coarse_index(predicted))

    session.close()

    Fine_classification(filepath, coarse_model_index(predicted), coarse_predict_result)



#网络构建以及训练， 测试， d代表大类的数字，00001-00006
def Fine_classification(filepath, d, coarse_predict_result):
    
    #图片的类别的数量
    n_classes = 43
    
    # Create a graph to hold the model. 创建Graphic
    #首先，我先创建一个Graph对象。TensorFlow有一个默认的全局图，但是我不建议使用它。设置全局变量通常是一个很坏的习惯，因为它太容易引入错误了。我更倾向于自己明确地创建一个图。
    graph = tf.Graph()
    
    # Create model in the graph.
    with graph.as_default():
        
        #images_ph的占位符形式大概是[None, 32, 32, 3]，这个分别表示[批次，高，宽，通道]
        #批次是None表示批次是灵活的，这就意味着我们可以在不改变代码的情况下修改批次。
        #请注意参数的顺序，因为在像NCHW这样的模型中，参数的顺序是不同的。
        #接下来，我定义了全连接层。与往常不同，我没有实现y = xW + b等式，我使用了一个简单的非线性函数来实现激活函数的功能。我期望输入是个1维的数组，所以首先我将图片平整化。
        #ReLU 作为激活函数。
        #所有负数的函数值都是0，这样对于分类任务和训练速度来说都会强于sigmoid和tanh函数
        
        # 设置占位符用来放置图片和标签，占位符石tensorflow从主程序中接受输入的方式
        # 在graph.as_default() 中创建的占位符（和其他所有操作）， 这样的好处是他们成为了创建图的一部分，而不是在全局图中
        #参数 images_ph 的维度是 [None, 32, 32, 3]，这四个参数分别表示 [批量大小，高度，宽度，通道] （通常缩写为 NHWC）。批处理大小用 None 表示，意味着批处理大小是灵活的，也就是说，我们可以向模型中导入任意批量大小的数据，而不用去修改代码。注意你输入数据的顺序，因为在一些模型和框架下面可以使用不同的排序，比如 NCHW。
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_ph = tf.placeholder(tf.int32, [None])
        
        #接下来，我定义一个全连接层，而不是实现原始方程 y = xW + b。在这一行中，我使用一个方便的函数，并且使用激活函数。模型的输入时一个一维向量，所以我先要压平图片。
        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)
        
        #全连接层的输出是一个长度是62的对数矢量（从技术上分析，它的输出维度应该是 [None, 62]，因为我们是一批一批处理的）。
        #输出的数据可能看起来是这样的：[0.3, 0, 0, 1.2, 2.1, 0.01, 0.4, ... ..., 0, 0]。值越高，图片越可能表示该标签。
        # Fully connected layer.
        # Generates logits of size [None, 62]
        # logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
        logits = LeNet(images_ph, n_classes)
        
        #在这个项目中，我们只需要知道最大值所对应的索引就行了，因为这个索引代表着图片的分类标签，这个求解最大操作可以如下表示：
        #argmax 函数的输出结果将是一个整数，范围是 [0, 61]。
        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        predicted_labels = tf.argmax(logits, 1)
        
        #我们需要将标签和神经网络的输出转换成概率向量。TensorFlow中有一个 sparse_softmax_cross_entropy_with_logits 函数可以实现这个操作。这个函数将标签和神经网络的输出作为输入参数，并且做三件事：第一，将标签的维度转换为 [None, 62]（这是一个0-1向量）；第二，利用softmax函数将标签数据和神经网络输出结果转换成概率值；第三，计算两者之间的交叉熵。这个函数将会返回一个维度是 [None] 的向量（向量长度是批处理大小），然后我们通过 reduce_mean 函数来获得一个值，表示最终的损失值。
        # Define the loss function.
        # Cross-entropy is a good choice for classification.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
        
        #下一个需要处理的就是选择一个合适的优化算法。我一般都是使用 ADAM 优化算法，因为它的收敛速度比一般的梯度下降法更快。如果你想知道不同优化器之间的比较结果，
        # Create training op.
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        
        #图中的最后一个节点是初始化所有的操作，它简单的将所有变量的值设置为零（或随机值）。
        # And, finally, an initialization op to execute before training.
        # TODO: rename to tf.global_variables_initializer() on TF 0.12.
        init = tf.initialize_all_variables()
        
        #模型保存加载工具
        saver = tf.train.Saver()
        
        #判断模型保存路径是否存在，不存在就创建
        if not os.path.exists(d+'/'):
            os.mkdir(d+'/')
            print()
            print("file created", d)
            print()
        
        #这是我们迭代训练模型的地方。在我们开始训练之前，我们需要先创建一个会话（Session）对象。
        
        #初始化
        session = tf.Session(graph=graph)
        if os.path.exists(d+'/checkpoint'): #判断模型是否存在
            saver.restore(session, d+'/model.ckpt') #存在就从模型中恢复变量
            print("model haved been loaded", d)
        else:
            _ = session.run([init])


    #评估
    # Load the test dataset.
    #pil 读取图片，并将图片转位numpy数组
    im = Image.open(filepath)
    im_array = np.array(im)
    
    #测试的输入，图片（numpy数组形式），label
    image=im_array
    label=int(43)
    
    # Transform the images, just like we did with the training set.
    test_images32 = [skimage.transform.resize(image, (32, 32))]
    
    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]
                            #显示到训练结果的文本框中
    fine_predict_result = predicted
    fine_result.set(predicted)
    fine_info.set(fine_index(predicted))
    
    print("predicted label is ", predicted)
    # Close the session. This will destroy the trained model.
    print()
    print("predicted info is ", fine_index(predicted))
    print("fine_predict_result",fine_predict_result)
    session.close()


global_conv2 = tf.zeros((1,5,5,16))
#粗分类lenet-5网络
def LeNet(x, n_classes):
    global global_conv2
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    global_conv2 = conv2
    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes （43 here, number of classes).
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    print("logits is ", logits)
    return logits



#ui控件
selectimg = Button(image_frame,text='Select image',anchor = 'center',command = callback)
selectimg.grid(row = 0, column = 0, sticky = NW, pady = 8, padx = 20)

cutimg1 = Button(image_frame,text='Cut image',anchor = 'center',command = show_image_detected)#调用裁剪的函数
cutimg1.grid(row = 0, column = 1, sticky = NW, pady = 8, padx = 20)

testimg2 = Button(image_frame,text='Test image',anchor = 'center',command =Coarse_classification)#调用识别的函数
testimg2.grid(row = 0, column = 2, sticky = NW, pady = 8, padx = 20)

nextimg3 = Button(image_frame,text='Next image',anchor = 'center',command = show_next_img)#调用下一张图片的函数
nextimg3.grid(row = 0, column = 3, sticky = NW, pady = 8, padx = 20)


totalimg= Entry(image_frame,textvariable = detected_img, width=10).grid(row = 1, column = 0)
#截取出来的全部图片

testedimage= Entry(image_frame,textvariable = showing_img, width=10).grid(row = 1, column = 1)
#剩余未被识别的图片

label1=Label(image_frame,text = 'original image:').grid(row = 3, column = 0)

label2=Label(image_frame,text = 'Detected image:').grid(row = 3, column = 1)

label2=Label(image_frame,text = 'coarse classification:').grid(row = 3, column = 2)

label2=Label(image_frame,text = 'fine classification:').grid(row = 3, column = 3)

coarseresult= Entry(image_frame,textvariable = coarse_result, width=10).grid(row = 4, column = 2)

coarseinfo= Entry(image_frame,textvariable = coarse_info, width=20).grid(row = 5, column = 2)

fineresult= Entry(image_frame,textvariable = fine_result, width=10).grid(row = 4, column = 3)

fineinfo= Entry(image_frame,textvariable = fine_info, width=20).grid(row = 5, column = 3)

'''
label2= Label(image_frame,text = "实际结果:").grid(row = 4, column = 0)
realresult= Entry(image_frame, textvariable = real_result, width=10).grid(row = 5, column = 1)

label3= Label(image_frame,text = "是否匹配:").grid(row = 6, column = 0)
checkresult= Entry(image_frame, textvariable = check_result, width=10).grid(row = 7, column = 1)
'''

image_frame.pack()
root.mainloop()


