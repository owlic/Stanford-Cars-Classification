import torch
import torch.nn as nn
import torchvision.models as models
import copy
import matplotlib.pyplot as plt
import numpy as np
from dataset import Image_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path


resnet101 = models.resnet101(pretrained=True)
fc_features = resnet101.fc.in_features
resnet101.fc = nn.Linear(fc_features, 196)

# REPRODUCIBILITY
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA_devices = 0
dataset_train_root = './cars_train_crop'
dataset_test_root = './cars_test_crop'
path_to_weights = './ResNet101_pretrained.pth'

num_epochs = 50
training_loss_values = []
training_acc_values = []
testing_acc_values = []


def test():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
    test_set = Image_Dataset(Path(dataset_test_root), data_transform)
    data_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True, num_workers=1)
    classes = [_dir.name for _dir in Path(dataset_test_root).glob('*')]

    model = torch.load(path_to_weights)
    model = model.cuda(CUDA_devices)
    model.eval()

    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_devices))
            labels = Variable(labels.cuda(CUDA_devices))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # total
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    testing_acc_values.append(float(f'{(total_correct / total):.4f}'))
    print('Accuracy on the ALL test images: %d %%' % (100 * total_correct / total))

    for i, c in enumerate(classes):
        print('Accuracy of %5s : %2d %%' % (c, 100 * class_correct[i] / class_total[i]))
    print('\n')


def train():
	data_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	train_set = Image_Dataset(Path(dataset_train_root), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)

	model = resnet101
	model = model.cuda(CUDA_devices)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.002, momentum=0.9)

	for epoch in range(1, num_epochs + 1, 1):
		print(f'Epoch: {epoch}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_devices))
			labels = Variable(labels.cuda(CUDA_devices))

			optimizer.zero_grad()

			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			# revise loss.data[0] --> loss.item()
			training_corrects += torch.sum(preds == labels.data)

		training_loss = training_loss / len(train_set)
		training_acc = training_corrects.double() / len(train_set)

		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

		training_loss_values.append(float(f'{training_loss:.4f}'))
		training_acc_values.append(float(f'{training_acc:.4f}'))

		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

		model.load_state_dict(best_model_params)
		torch.save(model, f'ResNet101_pretrained.pth')

		if (epoch % 10 == 0):
			test()


def plot():
	epoch_training = np.linspace(1, num_epochs, num_epochs)
	epoch_testing = np.linspace(10, num_epochs, int(num_epochs / 10))

	plt.plot(epoch_training, training_acc_values, label="training")
	plt.plot(epoch_testing, testing_acc_values, label="testing")
	plt.xlabel("epochs")
	plt.title("Accuracy (pretrained)")
	plt.legend(loc="best")
	plt.show()

	print("\n")

	plt.plot(epoch_training, training_loss_values)
	plt.xlabel("epochs")
	plt.title("loss (pretrained)")
	plt.show()

	print("\n")


if __name__ == '__main__':
	train()
	plot()
