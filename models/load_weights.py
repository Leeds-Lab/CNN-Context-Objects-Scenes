import os
import requests
import torch
import models.GRCNN as Grcnn
from torchvision import models
from zipfile import ZipFile

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
            GRCNN55: self.grcnn55()
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

    def grcnn55(self):
        grcnn55_ = Grcnn.grcnn55()
        model_file = self.get_grcnn_checkpoints()
        print(model_file)
        grcnn55_.load_state_dict(torch.load(model_file))
        grcnn55_.eval()
        return grcnn55_

    # AlexNet, ResNet18, and ResNet50 weight links can be found at https://github.com/CSAILVision/places365
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

    # From https://github.com/Jianf-Wang/GRCNN as GRCNN-55
    def get_grcnn_checkpoints(self):
        directory = './models/checkpoints/'
        model_file = 'checkpoint_params_grcnn55'
        path_name = directory + model_file + '.pt'

        # The GRCNN checkpoint is embedded in a zip file, which needs to be extracted from the remote source if a local copy is unavailable
        if not os.path.exists(path_name):
            zip_name = directory + model_file + '.zip'
            print(f"\nDownloading GRCNN55 zip file...")
            zip_url = 'https://drive.google.com/u/1/uc?id=12SusuxuMttubHIfNqn3gmEqwxLYXU_vZ&export=download&confirm=t&uuid=a13734ee-4da9-421b-b901-55ffed0d3664&at=ALAFpqxmr0Y_-fsxLsNf046vEw2F:1667576104138'
            grcnn_zip = requests.get(zip_url)
            with open(zip_name, 'wb') as f: f.write(grcnn_zip.content)
            print(f"Extracting checkpoint...\n")
            with ZipFile(zip_name, 'r') as zObject:
                zObject.extractall(path=directory)
            os.remove(zip_name)
            print(f'Done! Checkpoint saved in {directory}')
        return path_name
        