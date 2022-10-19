import numpy as np
import cv2
import math

def polling(pol_type,img,shape,Sp,P=(0,0)):
    M=img.shape[0]
    N=img.shape[1]
    C=img.shape[2]
    (Px,Py)=P

    padded_img=np.zeros((M+2*Px,N+2*Py,C))
    padded_img[Px:M+Px,Py:N+Py,:]=img[:,:,:]
    img=padded_img

    m=shape[0]
    n=shape[1]

    [Sx,Sy]=Sp

    M_o=math.ceil((M-m+2*Px)/Sx) +1
    N_o=math.ceil((N-n+2*Py)/Sy) +1

    op=np.zeros((M_o,N_o,C))

    for channel in range(C):
        x = 0
        y = 0
        for i in range(0, M - m, Sx):
            y = 0
            for j in range(0, N - n, Sy):
                if pol_type=='max':
                    op[x, y,channel] = np.max(img[i:i+m,j:j+n,channel])
                elif pol_type=='min':
                    op[x, y, channel] = np.min(img[i:i + m, j:j + n, channel])
                else:
                    op[x, y, channel] = np.mean(img[i:i + m, j:j + n, channel])

                y += 1
            x += 1
    print("polling:",op.shape)
    return op

def activation(img,a_f):
    op=0
    if a_f=='relu':
        mask=img>0
        op=np.multiply(mask,img)
        #print(op.shape)

    if a_f=='sigmoid':
        op=1/1+np.exp(-img)

    if a_f=='tanh':
        op=np.tanh(img)

    return op

def CNN(img,kernel,Sc,pol_type,Sp,a_f,P=(0,0),dim=(2,2)):
    #destructuring input
    [Sx,Sy]=Sc
    [Px,Py]=P
    M=img.shape[0]
    N=img.shape[1]
    C=img.shape[2]
    m=kernel.shape[0]
    n=kernel.shape[1]
    c=kernel.shape[2]

    #padding
    padded_img=np.zeros((M+2*Px,N+2*Py,C))
    padded_img[Px:Px+M,Py:Py+N,:]=img[:,:,:]
    img=padded_img
    #print("padded img:",img.shape)
    #output shape
    M_o=math.ceil((M-m+2*Px)/Sx) +1
    N_o=math.ceil((N-n+2*Py)/Sy) +1

    op=np.zeros((M_o,N_o,c))

    #actual convolution
    x=0
    y=0
    #print(len(range(0,img.shape[1]-n,Sy)))
    for k_channel in range(c):
        for i in range(0,img.shape[0]-m,Sx):
                y=0
                for j in range(0,img.shape[1]-n,Sy):
                    sum=0
                    for channel in range(C):
                        sum+=np.sum(np.multiply(img[i:i+m,j:j+n,channel],kernel[:,:,k_channel]))
                    if(x<op.shape[0] and y<op.shape[1]):
                        op[x,y,k_channel]=sum
                    y+=1
                x+=1

    print("CNN", op.shape)
    #activation and polling if needed
    if a_f!='':
        op=activation(op,a_f)
    if pol_type!='':
        op=polling(pol_type,op,dim,Sp)
    #cv2.imshow('op',op[:,:,0])
    #cv2.waitKey(0)

    return op

def inception(conv_prev,c1,c_reduced3,c_3,c_reduced5,c_5,c_pool):
    kernel_1 = np.ones((1, 1, c1))
    kernel_2a = np.ones((1, 1, c_reduced3))
    kernel_2b = np.random.randn(3, 3, c_3)
    kernel_3a = np.ones((1, 1, c_reduced5))
    kernel_3b = np.random.randn(5, 5, c_5)
    kernel_4 = np.ones((1, 1, c_pool))

    conv_1 = CNN(conv_prev, kernel_1, [1, 1], '', [], 'relu')
    conv_2a = CNN(conv_prev, kernel_2a, [1, 1], '', [], '')
    conv_2b = CNN(conv_2a, kernel_2b, [1, 1], '', [1, 1], 'relu',P=(1,1))
    conv_3a = CNN(conv_prev, kernel_3a, [1, 1], '', [1, 1], '')
    conv_3b = CNN(conv_3a, kernel_3b, [1, 1], '', [1, 1], 'relu',P=(2,2))
    conv_4a = polling('max', conv_prev, [3, 3], [1, 1],P=(1,1))
    conv_4b = CNN(conv_4a, kernel_4, [1, 1], '', [], 'relu')

    conv = np.append(conv_1, conv_2b, axis=2)
    conv = np.append(conv, conv_3b, axis=2)
    conv = np.append(conv, conv_4b, axis=2)
    print("inception:",conv.shape)
    return conv

def drop_out(conv_prev,per):
    m=conv_prev.shape[0]
    m_retained=round((1-per)*m)
    idx=sorted(np.random.sample(range(0,m),m_retained))
    return conv_prev[idx,:]

def softmax(layer):
    s=np.sum(np.exp(layer))
    return np.exp(layer)/s

def GoogleNet(img):
    #convolution + max polling
    kernel1=np.random.randn(7,7,64)
    conv1=CNN(img,kernel1,[2,2],'max',[2,2],'relu',P=(2,2))

    #convolution 3x3 reduce
    kernel2_reduce=np.ones((1,1,64))
    conv2=CNN(conv1,kernel2_reduce,[1,1],'',[],'')

    #convolution 3x3 + max polling
    kernel3=np.random.randn(3,3,192)
    conv3=CNN(conv2,kernel3,[1,1],'max',[2,2],'relu',P=(1,1))

    #inception (3a)
    conv4=inception(conv3,64,96,128,16,32,32)

    #inception (3b)
    conv5=inception(conv4,128,128,192,32,96,64)

    #max poll
    conv5=polling('max',conv5,[3,3],[2,2],P=(0,0))

    #inception (4a)
    conv6=inception(conv5,192,96,208,16,48,64)

    # inception (4b)
    conv7 = inception(conv6, 160, 112, 224, 24, 64, 64)

    # inception (4c)
    conv8 = inception(conv7, 128, 128, 256, 24, 64, 64)

    # inception (4d)
    conv9 = inception(conv8, 112, 144, 288, 32, 64, 64)

    # inception (4e)
    conv10 = inception(conv9, 256, 160, 320, 32, 128, 128)

    #max poll
    conv10 = polling('max', conv10, [3, 3], [2, 2])

    # inception (5a)
    conv11 = inception(conv10, 256, 160, 320, 32, 128, 128)

    # inception (5b)
    conv12 = inception(conv11, 384, 192, 384, 48, 128, 128)

    #avg poll
    conv12=polling('avg',conv12,[7,7],[1,1],P=(0,0))

    #flatten
    flat=np.reshape(conv12,(-1,1))

    #drop out
    drop=drop_out(flat,40)

    #soft max
    soft=softmax(drop)

    #class determination
    op_class=np.argmax(soft)

    #retrun
    return op_class


#print(round(1.5))
img=np.array(cv2.imread('test.jpg'))
img.resize((224,224,3))
print(img.shape)
op=GoogleNet(img)