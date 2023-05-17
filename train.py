import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWritter
from utils import save_checkpoint, load_checkpoint, print_examples

from loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose(
		[
			transforms.Resize((356,356)),
   			transforms.RandomCrop((299,299)),
   			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
		]
	)
    
    train_loader, dataset = get_loader(
		root_folder = "flickr8k/images",
		annotation_files = "flickr8k/captions.txt",
		transform = transform,
		num_workers = 4
 	)
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    load_model = False
    save_model = True
    
    #HyperParameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4
    num_epochs = 100
    num_layers = 1
 
 