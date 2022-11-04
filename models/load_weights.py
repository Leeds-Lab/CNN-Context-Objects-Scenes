import os
import requests
import torch
import models.GRCNN as Grcnn
from torchvision import models

# CNN Model Names
ALEXNET = "AlexNet"
ALEXNET_PLACES365 = "AlexNet_Places365"
VGG16 = "Vgg16"
VGG19 = "Vgg19"
RESNET18 = "ResNet18"
RESNET18_PLACES365 = "ResNet18_Places365"
RESNET50 = "ResNet50"
RESNET50_PLACES365 = "ResNet50_Places365"
RESNEXT50_32X4D = "Resnext50_32x4d"
RESNET101 = "ResNet101"
RESNET152 = "ResNet152"
GOOGLENET = "GoogLeNet"
GRCNN55 = "GRCNN55"

# Contains call functions for models. GRCNN55, and the Places365 trained models need .pt and .tar files containing weights loaded before use
class Models:
    def __init__(self):
        super(Models, self).__init__()
        self.shallow_model = {}
        self.deep_model = {}

    def load_pytorch_models(self):
        self.shallow_model = {
            ALEXNET: self.alexnet(),
            VGG16: self.vgg16(),
            VGG19: self.vgg19(),
            ALEXNET_PLACES365: self.alexnet_places365()
        }

        self.deep_model = {
            RESNET18: self.resnet18(),
            RESNET18_PLACES365: self.resnet18_places365(),
            RESNET50: self.resnet50(),
            RESNET50_PLACES365: self.resnet50_places365(),
            RESNEXT50_32X4D: self.resnext50_32x4d(),
            RESNET101: self.resnet101(),
            RESNET152: self.resnet152(),
            GOOGLENET: self.googlenet(),
            # GRCNN55: self.grcnn55() # uncomment when pretrained weights path is available
        }

    def alexnet(self):
        return models.alexnet(weights=True)

    def vgg16(self):
        return models.vgg16(weights=True)

    def vgg19(self):
        return models.vgg19(weights=True)

    def resnet18(self):
        return models.resnet18(weights=True)

    def resnet50(self):
        return models.resnet50(weights=True)

    def resnext50_32x4d(self):
        return models.resnext50_32x4d(weights=True)

    def resnet101(self):
        return models.resnet101(weights=True)

    def resnet152(self):
        return models.resnet152(weights=True)

    def googlenet(self):
        return models.googlenet(weights=True)

    def alexnet_places365(self):
        arch = 'alexnet'
        return self.get_pretrained_places_model(arch)

    def resnet18_places365(self):
        arch = 'resnet18'
        return self.get_pretrained_places_model(arch)

    def resnet50_places365(self):
        arch = 'resnet50'
        return self.get_pretrained_places_model(arch)

    def get_pretrained_places_model(self, arch):
        model_file = f'./models/tarballs/{arch}_places365.pth.tar'
        if not os.path.exists(model_file):
            print(f"\nDownloading {arch}_places365.pth.tar...")
            weight_url = f'http://places2.csail.mit.edu/models_places365/{arch}_places365.pth.tar'
            places365_data = requests.get(weight_url)
            with open(f'./models/tarballs/{arch}_places365.pth.tar', 'wb') as f: f.write(places365_data.content)
            print(f"Done! Pretrained weights saved in ./models/tarballs/\n")
        model_places365 = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model_places365.load_state_dict(state_dict)
        model_places365.eval()
        return model_places365

    # def grcnn55(self):
    #     grcnn55_ = Grcnn.grcnn55()
    #     grcnn55_.load_state_dict(torch.load('./models/checkpoints/checkpoint_params_grcnn55.pt'))
    #     grcnn55_.eval()
    #     return grcnn55_