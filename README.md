# separate-for-sensitive
 A pytorch-based deep learning project aimed at a rough classification of sensitive images (pornography, blood)  
This is a deep learning project that I have completed independently.   
I hope to show some examples of the project for students who also want to have learning ideas about machine learning but have no way to start.  

#Chinese：
==
这是我肚子完成的一个基于Pytorch的深度学习项目，旨在对敏感图片（色情、血腥）进行一个粗分类。  
如果你还没有过项目实战经验，可以浏览我的许多代码进行学习，很简单，并没有很复杂的特征识别，就是单纯的学习  
惊讶的是，我所使用的ResNet竟然真的将我所使用的数据集内的识别正确率提高到了大于80%  

#接下来讲一下main里相关函数作用：
--
ToTensor与Resize是重写函数，因为对dataset类的分类处理不好，所以重新定义了一下  
ReSize是将图片裁剪为与神经网络接口大小相同的图片  
ToTensor是torch里将图片的数据张量化为可供net训练的数据模型  

#几个模块的作用：  
--
url get：是最初对一个包含数据集url的txt文件解析并爬取图片文件本身的一个模块  
make dataset：是将图片信息存储到一个单独的文件供dataset类读取并载入图片文件  
TestModel:顾名思义就是专门用来检测训练后的网络参数是否能符合预期效果的模块，没什么好说的，需要载入Resize和ToTensor，因为还要将图片输入网络中  

#训练中一些算法及参数说明：  
--
优化器中：  
lr:就是训练步长，早期大，后期小，不在赘述，我的lr参数大致是从0.1一路降到最后的1e-6，是非常明显的区别，后期因为梯度值难以下降，所以lr的值一降再降(就是常说的α)  
weight_decay:正则化参数，由于网络中的乘算过多，为了防止参数在多层运算过程中变得过大或过小（梯度爆炸和梯度消失）而设置，这在实践中也十分明显，如不加以控制，基本连一次训练都完成不了  
最后，优化器选择的是公认的Adam算法，双参数调控可以更加精准。  

最最后，你在实践过程中可能会碰到的烦人的小问题：
--
'''
我甚至觉得我遇到的问题会是些常见问题  
1  
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=0)  
dataloader将image数据传入后最好仍用字典赋值  
我之前改过这个方法，但是网上太多都是直接赋值  
像这样  
inputs,labels = data
这样自定义数据集很可能出错（网上都是现成的torchvision数据集）  
2  
输入数据类型问题  
RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.FloatTensor) should be the same  
权重值希望的float32类型，在input到网络之前得个改类型  
我一度以为是我输入的Float64有问题，这个报错的语意真难琢磨  
3  
inputs = inputs.cuda()  
将数据改成cuda类型才能加入device，（很奇怪啊，label不变就能加入device）本来以为不加也行，网上各种实例都没加，但自定义数据集还是需要加上  
4  
batch_size = 16  
是我草率了，电脑8太行，改成32x32的图像后batch_size调成8也没问题，但问题是网络只兼容32x32的图对识别就很不友好，  
（这里插一下）  
RuntimeError: size mismatch, m1: [2 x 25088], m2: [512 x 10] at xxxxx(后面忘了)[我网上找的较规范的34层ResNet看来输入需要32x32]  
我试了试，以人的视觉是没什么问题，但我对网络不放心  
5  
最后就是这个网络的问题了，这是我最开始最不担心的问题  
唉，我再研究研究改改吧，虽然也不是不能看  
个人做自定义数据集真是过河都没石头摸  
'''
(问题记录摘自原数列学会社区blog，现以转入内网，黏贴出来供交流学习)
