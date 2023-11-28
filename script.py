import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImagePredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_path)
        self.transform = self.load_transform()

    def load_model(self, model_path):
        model = Net().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def load_transform(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform

    def get_image_from_url(self, image_url):
        import requests
        from PIL import Image
        from io import BytesIO

        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image

    def predict_image(self, image):
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        output = self.model(image)
        _, predicted = torch.max(output, 1)

        prob = F.softmax(output, dim=1)[0] * 100

        prob_res = round(prob[predicted[0]].item(), 2)

        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        return f'<span style="color: blue">Predicted: {classes[predicted[0]]}</span><br/>' \
               f'Percent: {prob_res}'

# pred = ImagePredictor('cifar_net.pth')
#
# image = pred.get_image_from_url('https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_3x4.jpg')
# print(pred.predict_image(image))
# image = pred.get_image_from_url('https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg')
# print(pred.predict_image(image))
# image = pred.get_image_from_url('https://e3.365dm.com/21/07/1600x900/skynews-boeing-737-plane_5435020.jpg?20210702173340')
# print(pred.predict_image(image))