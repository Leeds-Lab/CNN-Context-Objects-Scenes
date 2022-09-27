import torch
from torchvision import models
import models.GRCNN as Grcnn

# Contains call functions for models. GRCNN55, and the Places365 trained models need .pt and .tar files containing weights loaded before use
class Models:
    def alexnet():
        return models.alexnet(weights=True)

    def vgg16():
        return models.vgg16(weights=True)

    def vgg19():
        return models.vgg19(weights=True)

    def resnet18():
        return models.resnet18(weights=True)

    def resnet50():
        return models.resnet50(weights=True)

    def resnext50_32x4d():
        return models.resnext50_32x4d(weights=True)

    def resnet101():
        return models.resnet101(weights=True)

    def resnet152():
        return models.resnet152(weights=True)

    def googlenet():
        return models.googlenet(weights=True)

    def grcnn55():
        grcnn55_ = Grcnn.grcnn55()
        grcnn55_.load_state_dict(torch.load('./models/checkpoints/checkpoint_params_grcnn55.pt'))
        grcnn55_.eval()
        return grcnn55_

    def alexnet_places365():
        arch = 'alexnet'
        model_file = './models/tarballs/%s_places365.pth.tar' % arch
        alexnet_places365_ = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        alexnet_places365_.load_state_dict(state_dict)
        alexnet_places365_.eval()
        return alexnet_places365_

    def resnet18_places365():
        arch = 'resnet18'
        model_file = './models/tarballs/%s_places365.pth.tar' % arch
        resnet18_places365_ = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        resnet18_places365_.load_state_dict(state_dict)
        resnet18_places365_.eval()
        return resnet18_places365_

    def resnet50_places365():
        arch = 'resnet50'
        model_file = './models/tarballs/%s_places365.pth.tar' % arch
        resnet50_places365_ = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        resnet50_places365_.load_state_dict(state_dict)
        resnet50_places365_.eval()
        return resnet50_places365_