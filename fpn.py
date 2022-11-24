class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        #构建C1
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
 
        #自下而上搭建C2、C3、C4、C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        #对C5减少通道，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
 
        #3*3卷积融合
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
 
        #横向连接，保证每一层通道数一致
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d( 512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d( 256, 256, 1, 1, 0)
 
    #构建C2到C5
    def _make_layer(self, planes, blocks, stride=1):
        downsample  = None
        #如果步长不为1，进行下采样
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample  = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        #更新输入输出层
        self.inplanes = planes * Bottleneck.expansion
        #根据block数量添加bottleneck的数量
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers
                             
    #自上而下上采样
    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        #逐个元素相加
        return F.upsample(x, size=(H,W), mode='bilinear') + y
 
    def forward(self, x):
        #自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
 
        #自上而下，横向连接
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        
        #卷积融合，平滑处理
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5
 