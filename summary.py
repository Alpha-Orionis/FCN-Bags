from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
import torch
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 当有GPU时部署到GPU没有则只用CPU

vgg_model = VGGNet(requires_grad=True, show_params=False)  # 使用在另一个py文件中的VGG网络模型
fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)

fcn_model = fcn_model.to(device)  # 将模型部署到设备上

summary(fcn_model, (3, 128, 128))

