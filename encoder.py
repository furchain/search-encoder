import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from sklearn.preprocessing import normalize
import warnings
from threading import Lock
from io import BytesIO

warnings.filterwarnings("ignore")


class Encoder:

    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.layer = self.model._modules.get('avgpool')
        self.model.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.lock = Lock()

    def get_vector(self, image_bytes) -> (BaseException, np.ndarray):
        # 1. Load the image with Pillow library
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            # 2. Create a PyTorch Variable with the transformed image
            t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        except BaseException as e:
            return e, None
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(1, 512, 1, 1)

        # 4. Define a function that will copy the output of a layer
        def copy_data(_, __, o):
            my_embedding.copy_(o.data)

        # 5. Attach that function to our selected layer
        with self.lock:
            try:
                h = self.layer.register_forward_hook(copy_data)
                # 6. Run the model on our transformed image
                self.model(t_img)
            except BaseException as e:
                return e, None
            # 7. Detach our copy function from the layer
            finally:
                try:
                    h.remove()
                except NameError or UnboundLocalError:
                    pass
        # 8. Return the normalized feature vector
        return None, normalize(my_embedding.reshape(1, -1).numpy())

    @staticmethod
    def normalize(vec):
        return normalize(vec.reshape(1, -1))
