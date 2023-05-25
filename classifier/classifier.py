import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torchvision import transforms,datasets
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--do_training',action="store_true",help="Want to train or not")
do_training=parser.parse_args().do_training 



weight_path="./vgg19.pth"



vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)



for param in vgg19.parameters():
    param.requires_grad = False


num_classes = 2  
vgg19.classifier[6] = nn.Linear(4096, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg19.classifier.parameters(), lr=0.001, momentum=0.9)

if not do_training:
    vgg19.load_state_dict(torch.load(weight_path))


data_root = './dataset/'


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                            std=[0.229, 0.224, 0.225])])


train_dataset = datasets.ImageFolder(root=data_root+'train',transform=transform)


val_dataset = datasets.ImageFolder(root=data_root+'val',transform=transform)


batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=0)


# print(train_dataset.classes)
# print(train_dataset.targets)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19 = vgg19.to(device)

if do_training:
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = vgg19(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        # Print average loss for each epoch
        print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader)}")
        torch.save(vgg19.state_dict(),weight_path)
        


# Evaluation
correct = 0
total = 0
vgg19.eval()

with torch.no_grad():
    for images, labels in val_loader:

        print(labels)
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = vgg19(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        # print("Prediction : {}  ||||| Original : {}".format(predicted,labels))
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy}%")
