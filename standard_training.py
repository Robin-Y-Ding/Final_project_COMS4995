from model import ModelA
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from  utils import draw_tsne

batch_size = 100
input_dim = 784
output_dim = 10

def dataloader():
    # load the dataset from pytorch official MNIST dataset; get the training data first
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    # load the dataset from pytorch official MNIST dataset; get the test data first
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    # split the training dataset to training and valid
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) - int(len(train_dataset) * 0.9)])
    # Create training loader from the training data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Create the valid loader from the valid data
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    # Create the test loader from the test loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def dataloader_vis():
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5000, shuffle=False)
    return train_loader


def train(train_loader, valid_loader):
    # Create the loss function for further use: in the "torch.nn.CrossEntropyLoss()", the softmax and the nlloss has both included
    # so we will not do the softmax anymore in the "forward" function in each class
    lossFunc = torch.nn.CrossEntropyLoss()

    # indicate the learning rate
    lr_rate = 5.0 * 1e-4
    # initialize the model
    model = ModelA()
    # Use Adam for Softmax regression model, by following the tf example.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    train_on_gpu = torch.cuda.is_available()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to('cuda')
    else:
        device = torch.device("cpu")
    minibatch = 0
    # best_loss = float("inf")
    best_acc = float("-inf")
    # use the same iteration as the example
    while minibatch <= 30000:
        for images, labels in train_loader:

            # transfer the data to pytorch Variable so that the model can handel
            images = Variable(images.view(28 * 28, -1))
            labels = Variable(labels)
            images, labels = images.cuda(), labels.cuda()
            # zero out the gradients for each batch
            optimizer.zero_grad()
            # Reshape image for convolutional inputs
            images = images.reshape(-1, 1, 28, 28)
            # pass the data into the model
            outputs = model(images)
            # Compute the loss;
            loss = lossFunc(outputs, labels)
            # Back propagation
            loss.backward()
            # Optimize the weights by using stochastic gradient descent
            optimizer.step()

            minibatch += 1
            if minibatch % 1000 == 0:
                # calculate Accuracy
                correct = 0
                total = 0
                valid_loss = 0
                valid_batch = 0
                # load the valid data
                for images, labels in valid_loader:
                    valid_batch += 1
                    # transfer the data to pytorch Variable so that the model can handle
                    images = Variable(images.view(28 * 28, -1))
                    # Reshape image for convolutional inputs
                    images, labels = images.cuda(), labels.cuda()
                    images = images.reshape(-1, 1, 28, 28)
                    # change the model to eval mode, so that nothing will be dropped
                    model.eval()
                    # pass the valid data
                    outputs = model(images)
                    # please note that the loss will be cumulative batch by batch during validation
                    valid_loss += lossFunc(outputs, labels).item()
                    # torch.max will return a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim.
                    # here we just need the indices
                    _, predicted = torch.max(outputs.data, 1)
                    # get the total number of samples in this batch
                    total += labels.size(0)
                    # cumulatively get the correct predictions
                    correct += (predicted == labels).sum()
                # calculate the loss
                valid_loss = valid_loss / valid_batch
                # return back to train status, so that we can keep using dropout for next training steps
                model.train()
                # calculate the accuracy
                accuracy = 100.0 * correct/total
                print('Evaluating: Minibatch: {0}. Loss: {1}. Accuracy: {2:.2f}'.format(minibatch, valid_loss, accuracy))
                # if valid_loss < best_loss:
                if accuracy > best_acc:
                    # update the best acc so far
                    # best_loss = valid_loss
                    best_acc = accuracy
                    # save the model with best acc
                    torch.save(model.state_dict(), 'best_model_CNN.pth')
                    # save the optimizer for further training; this may not be used for grading
                    torch.save(optimizer, 'optimizer_CNN.pth')
                    print("Saving best model so far to best_model_CNN.pth")

def test(test_loader):
    # initialize a model for testing
    model_test = ModelA()
    # load the saved best model
    print("Restoring model from best_model_CNN.pth")
    model_test.load_state_dict(torch.load('best_model_CNN.pth'))
    # change the model to eval mode, to use all the weights without drop out
    model_test.eval()
    # no grads or back prop will be performed
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
            # transfer the data to pytorch Variable so that the model can handle
            images = Variable(images.view(28 * 28, -1))
            # Reshape image for convolutional inputs
            images = images.reshape(-1, 1, 28, 28)
            # pass the test data
            outputs = model_test(images)
            # torch.max will return a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim.
            # here we just need the indices
            _, predicted = torch.max(outputs.data, 1)
            # get the total number of samples in this batch
            total += labels.size(0)
            # cumulatively get the correct predictions
            correct += (predicted == labels).sum()
        accuracy = 100.0 * correct / total
        print("Testing: Accuracy: {0:.2f}.".format(accuracy))

def main():
    train_loader, valid_loader, test_loader = dataloader()
    # Example: How to use tSNE
    # visual_loader = dataloader_vis()
    # train_dataset_array = next(iter(visual_loader))[0].numpy().reshape(-1, 28 * 28)
    # training_labels_array = next(iter(visual_loader))[1].numpy()
    # draw_tsne(train_dataset_array, training_labels_array)
    train(train_loader, valid_loader)
    test(test_loader)

if __name__ == '__main__':
    main()
