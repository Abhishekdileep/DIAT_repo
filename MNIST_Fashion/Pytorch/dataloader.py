import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import LeNet

from torchsummaryX import summary as summaryX
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.01
num_epochs = 10

train_dataset = datasets.MNIST(root='dataset/', train=True, 
                               transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, 
                              transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


from IPython.display import HTML, display
class ProgressMonitor(object):
    """
    Custom IPython progress bar for training
    """
    
    tmpl = """
        <p>Loss: {loss:0.4f}   {value} / {length}</p>
        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>
    """
 
    def __init__(self, length):
        self.length = length
        self.count = 0
        self.display = display(self.html(0, 0), display_id=True)
        
    def html(self, count, loss):
        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))
        
    def update(self, count, loss):
        self.count += count
        self.display.update(self.html(self.count, loss))

def train_new(model,criterion,optimizer,num_epochs,dataloaders,dataset_sizes,first_epoch=1):
  since = time.time() 
  best_loss = 999999
  best_epoch = -1
  last_train_loss = -1
  plot_train_loss = []
  plot_valid_loss = []
 
 
  for epoch in range(first_epoch, first_epoch + num_epochs):
      print()
      print('Epoch', epoch)
      running_loss = 0.0
      valid_loss = 0.0
      
      # train phase
      model.train()
 
      # create a progress bar
      progress = ProgressMonitor(length=dataset_sizes["train"])
 
      for data in dataloaders[0]:
          # Move the training data to the GPU
          inputs, labels  = data
          batch_size = inputs.shape[0]
        
          inputs = Variable(inputs.to(device))
          labels = Variable(labels.to(device))
 
          # clear previous gradient computation
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
 
          loss.backward()
          optimizer.step()
                      
          running_loss += loss.data * batch_size
          # update progress bar
          progress.update(batch_size, running_loss)
 
      epoch_loss = running_loss / dataset_sizes["train"]
      print('Training loss:', epoch_loss.item())
      writer.add_scalar('Training Loss', epoch_loss, epoch)
      plot_train_loss.append(epoch_loss)
 
      # validation phase
      model.eval()
      # We don't need gradients for validation, so wrap in 
      # no_grad to save memory
      with torch.no_grad():
        for data in dataloaders[-1]:
            inputs, labels  = data
            batch_size = inputs.shape[0]
 
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            outputs = model(inputs)
 
            # calculate the loss
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            
            # update running loss value
            valid_loss += loss.data * batch_size
 
      epoch_valid_loss = valid_loss / dataset_sizes["test"]
      print('Validation loss:', epoch_valid_loss.item())
      plot_valid_loss.append(epoch_valid_loss)
      writer.add_scalar('Validation Loss', epoch_valid_loss, epoch)
    
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
 
  return plot_train_loss, plot_valid_loss, model

if __name__=="__main__":
  train_losses, valid_losses, model = train_new(model = model ,criterion = criterion,optimizer = optimizer,
                                                num_epochs=20,dataloaders = [train_loader, test_loader],
                                                dataset_sizes = dataset_sizes)
