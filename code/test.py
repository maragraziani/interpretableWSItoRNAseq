import sys, getopt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.utils.data
from sklearn import metrics 
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree, cKDTree
from sklearn.cluster import MiniBatchKMeans, KMeans, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics 
from scipy.stats import entropy
import argparse
import warnings
warnings.filterwarnings("ignore")
import setproctitle


argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("working on gpu")
else:
    device = torch.device("cpu")
    print("working on cpu")
print(torch.backends.cudnn.version())

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-c', '--CNN', help='cnn architecture to use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-p', '--pool', help='pooling algorithm',type=str, default='att')
parser.add_argument('-g', '--gene', help='gene name',type=str, default='AXIN2')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-t', '--TASK', help='task (binary/multilabel/geneExp)',type=str, default='multilabel')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=True)
parser.add_argument('-i', '--input_folder', help='path of the folder where train.csv and valid.csv are stored',type=str, default='./partition/')
parser.add_argument('-o', '--output_folder', help='path where to store the model weights',type=str, default='./models/')
parser.add_argument('-w', '--wsi_folder', help='path where WSIs are stored',type=str, default='./images/')

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = args.pool
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
TASK = args.TASK
EMBEDDING_bool = args.features
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder
WSI_FOLDER = args.wsi_folder
GENE =args.gene
PRE_TRAIN_LOG_FILE_VAL_ACC = os.path.join(OUTPUT_FOLDER,'eval_test.csv')
PRE_ATTENTION_VAL = os.path.join(OUTPUT_FOLDER,'attention_test.csv')
EXPERIMENT_TYPE="T_{}_{}_{}".format(GENE,CNN_TO_USE,pool_algorithm)
PATH_TO_TRAINED_MODEL = "{}/MIL_colon_geneExp.pt".format(OUTPUT_FOLDER)
if pool_algorithm == 'baseline':
    PATH_TO_TRAINED_MODEL = "{}/MIL_colon_geneExp_temporary.pt".format(OUTPUT_FOLDER)

setproctitle.setproctitle('{}'.format(EXPERIMENT_TYPE))
seed = 0

torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("TASK: " + str(TASK))
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("POOLING ALGORITHM: " + str(pool_algorithm))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))

#create folder (used for saving weights)
def create_dir(models_path):
	if not os.path.isdir(models_path):
		try:
			os.mkdir(models_path)
		except OSError:
			print ("Creation of the directory %s failed" % models_path)
		else:
			print ("Successfully created the directory %s " % models_path)

def select_parameters_colour():
	hue_min = -15
	hue_max = 8

	sat_min = -20
	sat_max = 10

	val_min = -8
	val_max = 8

	p1 = np.random.uniform(hue_min,hue_max,1)
	p2 = np.random.uniform(sat_min,sat_max,1)
	p3 = np.random.uniform(val_min,val_max,1)
	
	return p1[0],p2[0],p3[0]

def generate_transformer(prob = 0.5):
	list_operations = []
	probas = np.random.rand(4)
	
	if (probas[0]>prob):
		#print("VerticalFlip")
		list_operations.append(A.VerticalFlip(always_apply=True))
	if (probas[1]>prob):
		#print("HorizontalFlip")
		list_operations.append(A.HorizontalFlip(always_apply=True))
	if (probas[2]>prob):
		#print("RandomRotate90")
		list_operations.append(A.RandomRotate90(always_apply=True))
	if (probas[3]>prob):
		#print("HueSaturationValue")
		p1, p2, p3 = select_parameters_colour()
		#list_operations.append(A.HueSaturationValue(always_apply=True,hue_shift_limit=(p1,p1+1e-4),sat_shift_limit=(p2,p2+1e-4),val_shift_limit=(p3,p3+1e-4)))
		
	pipeline_transform = A.Compose(list_operations)
	return pipeline_transform


def generate_list_instances(filename):
# Mara: cambia questo per unire i CSV
	instance_dir = WSI_FOLDER
	fname = os.path.split(filename)[-1]
	
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'
	
	return instance_csv 


#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")

#create_dir(OUTPUT_FOLDER)

models_path = OUTPUT_FOLDER
checkpoint_path = models_path+'checkpoints_MIL/'
#create_dir(checkpoint_path)
patch_preds_dir = models_path+'patch_predictions/'
create_dir(patch_preds_dir)
#path model file
#model_weights_filename = models_path+'MIL_colon_'+TASK+'.pt'
#model_weights_filename_temporary = models_path+'MIL_colon_'+TASK+'_temporary.pt'

#CSV LOADING
print("CSV LOADING ")
csv_folder = INPUT_FOLDER

if (TASK=='binary'):

	N_CLASSES = 1
	#N_CLASSES = 2

	if (N_CLASSES==1):
		csv_filename_training = csv_folder+'train_binary.csv'
		csv_filename_validation = csv_folder+'valid_binary.csv'
		print("training file: ",csv_filename_training)
        
elif (TASK=='geneExp'):
	N_CLASSES = 1
	#N_CLASSES = 2
	print("Gene Expression Regression of gene {}".format(args.gene))
	if (N_CLASSES==1):
		csv_filename_training = csv_folder+'{}_test.csv'.format(args.gene)
		csv_filename_validation = csv_folder+'{}_test.csv'.format(args.gene)
		print("training file: ",csv_filename_training)
	
elif (TASK=='multilabel'):

	N_CLASSES = 5

	csv_filename_training = csv_folder+'train_multilabel.csv'
	csv_filename_validation = csv_folder+'valid_multilabel.csv'

	
#read data
train_dataset = pd.read_csv(csv_filename_training, sep=',', header=None).values#[:10]
valid_dataset = pd.read_csv(csv_filename_validation, sep=',', header=None).values#[:10]

class ImbalancedDatasetSampler_multilabel(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
		
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			for l in label:
				if l in label_to_count:
					label_to_count[l] += 1
				else:
					label_to_count[l] = 1
	
		# weight for each sample
		weights = []

		for idx in self.indices:
			c = 0
			for l in self._get_label(dataset, idx):
				c = c+(1/label_to_count[l])
			weights.append(c)
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		labels = np.where(dataset[idx,1:]==1)[0]
		#labels = dataset[idx,2]
		return labels
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

class ImbalancedDatasetSampler_single_label(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
			
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
				
		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset[idx,1]
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples


#MODEL DEFINITION
pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)

if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features

class MIL_model(torch.nn.Module):
	def __init__(self):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(MIL_model, self).__init__()
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		#self.conv_layers = siamese_model.conv_layers
		"""
		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)
		"""
		self.fc_feat_in = fc_input_features
		self.N_CLASSES = N_CLASSES
		
		if (EMBEDDING_bool==True):

			if ('resnet18' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			elif ('resnet34' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES
				#self.K = 1
			elif ('resnet50' in CNN_TO_USE):
				self.E = 256
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES

			#self.embedding = siamese_model.embedding
			self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
			self.embedding_fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)

		else:
			self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)
			
			if ('resnet18' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet34' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.L = self.E
				self.D = 256
				self.K = self.N_CLASSES

		if (pool_algorithm=='att'):

			self.attention = torch.nn.Sequential(
				torch.nn.Linear(self.L, self.D),
				torch.nn.Tanh(),
				torch.nn.Linear(self.D, self.K)
			)
		if (pool_algorithm=='baseline'):            
			self.mlp = torch.nn.Sequential(
				torch.nn.Linear(self.L, self.L//2),
				torch.nn.ReLU(),
				torch.nn.Linear(self.L//2, self.D),
				torch.nn.ReLU(),
				torch.nn.Linear(self.D, self.K),
			)
	
	def forward(self, x, conv_layers_out, labels_wsi_np):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		#if used attention pooling
		A = None
		#m = torch.nn.Softmax(dim=1)
		m_binary = torch.nn.Sigmoid() # Mara: this must be removed to obtain a MSE-based regression
		m_multiclass = torch.nn.Softmax()
		dropout = torch.nn.Dropout(p=0.2)

		self.labels = labels_wsi_np

		if x is not None:
			#print(x.shape)
			conv_layers_out=self.conv_layers(x)
			#print(x.shape)
			
			conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

		#print(conv_layers_out.shape)

		if ('mobilenet' in CNN_TO_USE):
			dropout = torch.nn.Dropout(p=0.2)
			conv_layers_out = dropout(conv_layers_out)
		#print(conv_layers_out.shape)

		if (EMBEDDING_bool==True):
			embedding_layer = self.embedding(conv_layers_out)
			features_to_return = embedding_layer

			embedding_layer = dropout(embedding_layer)
			logits = self.embedding_fc(embedding_layer) # Mara: so yes, the logits here are what you care about

		else:
			logits = self.fc(conv_layers_out)
			features_to_return = conv_layers_out

		#print(output_fcn.shape)
		if (TASK=='binary' and N_CLASSES==1):
			output_fcn = m_binary(logits)
		elif (TASK=='geneExp' and N_CLASSES==1):
			output_fcn = logits 
		else:
			output_fcn = m_multiclass(logits)

		output_fcn = torch.clamp(output_fcn, -10, 10) #Mara: why this?
		#print(output_fcn.size())
		if (pool_algorithm=='baseline'):
			# here you want to add the MLP layers output_pool=output_fcn
			A = self.mlp(features_to_return)
			output_pool = A
		if (pool_algorithm=='max'):
			output_pool = output_fcn.max(dim = 0)[0]
		elif (pool_algorithm=='avg'):
			output_pool = output_fcn.mean(dim = 0)
			#print(output_pool.size())
		elif (pool_algorithm=='lin'):
			output_pool = (output_fcn * output_fcn).sum(dim = 0) / output_fcn.sum(dim = 0)
			#print(output_pool.size())
		elif (pool_algorithm=='exp'):
			output_pool = (output_fcn * output_fcn.exp()).sum(dim = 0) / output_fcn.exp().sum(dim = 0)
			#print(output_pool.size())
		elif (pool_algorithm=='att'):

			if (EMBEDDING_bool==True):
				A = self.attention(features_to_return)
			else:
				A = self.attention(conv_layers_out)  # NxK
				
			#print(A.size())
			#print(A)
			A = F.softmax(A, dim=0)  # softmax over N #Mara: Eq. 8 in paper implements a softmax
			#print(A.size())
			#print(A)
			#A = A.view(-1, A.size()[0])
			#print(A)

			output_pool = (output_fcn * A).sum(dim = 0) / (A).sum(dim = 0)
			#print(output_pool.size())
			#print(output_pool)
			output_pool = torch.clamp(output_pool, -10, 10)
				
		return output_pool, output_fcn, A, features_to_return

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MIL_model()
model.to(device)

#load model weights
#model
model = torch.load(PATH_TO_TRAINED_MODEL)
#import pdb; pdb.set_trace()

from torchvision import transforms
prob = 0.5
pipeline_transform = A.Compose([
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	#A.ElasticTransform(alpha=0.1,p=prob),
	#A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-30,20),val_shift_limit=(-15,15),p=prob),
	])

preprocess = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, partition, pipeline_transform):
		self.list_IDs = list_IDs
		self.set = partition
		self.pipeline_transform = pipeline_transform

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index][0]
		# Load data and get label
		X = Image.open(ID)
		X = np.asarray(X)

		if (self.set == 'train'):
			#data augmentation
			X = self.pipeline_transform(image=X)['image']
			#X = pipeline_transform(image=X)['image']

		#data transformation
		input_tensor = preprocess(X).type(torch.FloatTensor)
				
		#return input_tensor
		return ID, input_tensor
	
class Dataset_bag(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		instances_filename = generate_list_instances(ID)
		y = self.labels[index]
		if (TASK=='binary' or TASK=='geneExp' and N_CLASSES==1):
			y = np.asarray(y)
		else:
			y = torch.tensor(y.tolist() , dtype=torch.float32)

				
		return instances_filename, y



batch_size_bag = 1

if (TASK=='binary' and N_CLASSES==1):
	sampler = ImbalancedDatasetSampler_single_label
	params_train_bag = {'batch_size': batch_size_bag,
		  'sampler': sampler(train_dataset)}
		  #'shuffle': True}
elif (TASK=='geneExp' and N_CLASSES==1):
	#sampler = FrozenSlidesSampler_single_label
	params_train_bag = {'batch_size': batch_size_bag,
		  #'sampler': sampler(train_dataset)}
		  'shuffle': True}

elif (TASK=='multilabel'):
	sampler = ImbalancedDatasetSampler_multilabel
	params_train_bag = {'batch_size': batch_size_bag,
		  'sampler': sampler(train_dataset)}
		  #'shuffle': True}

params_valid_bag = {'batch_size': batch_size_bag,
		  'shuffle': True}


num_epochs = EPOCHS

if (TASK=='binary' and N_CLASSES==1):
	training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1])
	training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

	validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1])
	validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)
if (TASK=='geneExp' and N_CLASSES==1):   
	training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1])
	training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

	validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1])
	validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)
else:
	training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1:])
	training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

	validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1:])
	validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

def compute_l2_loss(w):
      return torch.square(w).sum()

if (TASK=='geneExp'):
	criterion_wsi = torch.nn.MSELoss() #+ compute_l2_loss(model.parameters())# if p.requires_grad])
else:
	criterion_wsi = torch.nn.BCELoss()    

import torch.optim as optim
#optimizer_str = 'adam'
optimizer_str = 'sgd'

lr_str = '0.01'
lr_str = '0.001'
#lr_str = '0.0001'
#lr_str = '0.00001'

_decay_str = '0.0'
#wt_decay_str = '0.1'
#wt_decay_str = '0.05'
wt_decay_str = '0.01'
#wt_decay_str = '0.001'

lr = float(lr_str)
wt_decay = float(wt_decay_str)

if (optimizer_str == 'adam'):
	optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=False)
elif (optimizer_str == 'sgd'):
	optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=wt_decay, nesterov=True)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def accuracy_micro(y_true, y_pred):
    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()
    
    return metrics.accuracy_score(y_true_flatten, y_pred_flatten)

    
def accuracy_macro(y_true, y_pred):
    
    n_classes = len(y_true[0])
    
    acc_tot = 0.0
    
    for i in range(n_classes):
        
        acc = metrics.accuracy_score(y_true[i,:], y_pred[i,:])
        #print(acc)
        acc_tot = acc_tot + acc
        
    acc_tot = acc_tot/n_classes
    
    return acc_tot


def evaluate_validation_set(val_file_log, model, epoch, generator, attention_file_name=PRE_ATTENTION_VAL):
	#accumulator for validation set
	y_pred_val = []
	y_true_val = []

	valid_loss = 0.0

	mode = 'valid'
	wsi_store_loss = 0.0
	patches_store_loss = 0.0
	i_p = 0
	bool_patches = False

	filenames_wsis = []
	pred_cancers = []
	pred_hgd = []
	pred_lgd = []
	pred_hyper = []
	pred_normal = []

	model.eval()

	iterations = len(valid_dataset)
	attention_file = open(attention_file_name, 'w')
	with torch.no_grad():
		j = 0
		for inputs_bag,labels in generator:
			print('Epoch: [%d] (%d / %d) ' % (epoch, j, iterations))
				#inputs: bags, labels: labels of the bags
			labels_np = labels.cpu().data.numpy()
			len_bag = len(labels_np)

				#list of bags 
			if (TASK=='binary') or (TASK=='multiclass'):   
				filename_wsi = os.path.split(inputs_bag[0])[1]
				print("inputs_bag " + str(filename_wsi)) 
				inputs_bag = list(inputs_bag)
			filename_wsi = os.path.split(inputs_bag[0])[1]
			case_name = inputs_bag[0].split('/')[-2][:12]
			if (TASK=='geneExp'):
				case_name = inputs_bag[0].split('/')[-2][:12]
				inputs_bag = case_files[case_name]
				print("inputs_bag ", inputs_bag)

			for b in range(len_bag):
				labs = []
				labs.append(labels_np[b])
				labs = np.array(labs).flatten()

				labels = torch.tensor(labs).float().to(device)
				labels_wsi_np = labels.cpu().data.numpy()
				if (TASK=='geneExp'): 
					if len(inputs_bag[1])<2:
						try:
							file_path_ts = os.path.join(WSI_FOLDER, inputs_bag[1][0])#, '_paths_densely.csv')
							file_path_bs = None
						except:
							file_path_bs = os.path.join(WSI_FOLDER, inputs_bag[1][1])#, '_paths_densely.csv')
							file_path_ts = None
					else:
						file_path_ts = os.path.join(WSI_FOLDER, inputs_bag[1][0])
						file_path_bs = os.path.join(WSI_FOLDER, inputs_bag[1][1])#, '_paths_densely.csv')  
					csv_instances_ts =  csv_instances_bs = None   
					if file_path_ts:
						file_path_ts = file_path_ts + '/' + inputs_bag[1][0] + '_paths_densely.csv'
						csv_instances_ts = pd.read_csv(file_path_ts, sep=',', header=None)
					if file_path_bs:
						file_path_bs = file_path_bs + '/' + inputs_bag[1][1] + '_paths_densely.csv'
						csv_instances_bs = pd.read_csv(file_path_bs, sep=',', header=None)
					if file_path_bs and file_path_ts:
						all_csv_instances= pd.concat([csv_instances_ts, csv_instances_bs], ignore_index=True).values
					else: 
						all_csv_instances = csv_instances_ts.values if csv_instances_ts is not None else csv_instances_bs.values
					csv_instances=all_csv_instances
				else:         
					csv_instances = pd.read_csv(inputs_bag[b], sep=',', header=None).values
					#number of instances
				n_elems = len(csv_instances)
				print("num_instances " + str(n_elems))
					#params generator instances
				i=0#import pdb; pdb.set_trace()
				for i in range(n_elems):#csv_instances:
					elem_=csv_instances[i].item()
					file_name=str(elem_).split('/magnification_10x')[1]
					new_elem=WSI_FOLDER + file_name[1:]
					csv_instances[i]=new_elem#def_csv_instances.append([new_elem])
				#import pdb; pdb.set_trace()#params generator instances
				batch_size_instance = BATCH_SIZE
				num_workers = 4
				params_instance = {'batch_size': batch_size_instance,
						'shuffle': True,
						'num_workers': num_workers}

					#generator for instances
				instances = Dataset_instance(csv_instances,'valid',pipeline_transform)
				validation_generator_instance = data.DataLoader(instances, **params_instance)
				
				features = []
				list_ids = []
				with torch.no_grad():
					for ID, instances in validation_generator_instance:
						instances = instances.to(device)
						#import pdb; pdb.set_trace()

						# forward + backward + optimize
						feats = model.conv_layers(instances)
						feats = feats.view(-1, fc_input_features)
						feats_np = feats.cpu().data.numpy()
						
						features = np.append(features,feats_np)
						list_ids.append(ID)
				#import pdb; pdb.set_trace()
				list_ids=np.concatenate([list for list in list_ids])
				features_np = np.reshape(features,(n_elems,fc_input_features))
				
				del features, feats
				
				inputs = torch.tensor(features_np, requires_grad=True).float().to(device)
			
				predictions, probs, attn_layer, embeddings = model(None, inputs, labels_wsi_np)
				
				patch_preds_file=open(f"{patch_preds_dir}/{list_ids[0][51:63]}-preds.csv", 'w')
				for idx in range(len(list_ids)):
					patch_preds_file.write(f"{list_ids[idx], probs[idx][0]}\n")   
				patch_preds_file.close()
				i=0
				for item in list_ids:
					attention_file.write('{}, {}\n'.format(item, attn_layer[i]))
					i+=1
				loss = criterion_wsi(predictions, labels)
								
				#loss.backward() 

				outputs_wsi_np = predictions.cpu().data.numpy()

				del probs, attn_layer, embeddings
				if (pool_algorithm=='baseline'):
					valid_loss = wsi_store_loss 
					all_vals = []
					outs=outputs_wsi_np
					np.sort(outs)
					all_vals.append(np.mean(outs[:]))
					all_vals.append(np.mean(outs[:np.min((len(outs),10))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),20))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),50))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),100))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),200))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),500))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),1000))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),2000))]))
					all_vals.append(np.mean(outs[:np.min((len(outs),5000))]))
					pred=np.mean(np.asarray(all_vals))
					#import pdb; pdb.set_trace()#np.mean(np.asarray(all_vals))
					print("[debug] HERE: ",all_vals, np.mean(outs), pred)
					print('mean(output wsi): '+str(pred)+', label: '+ str(labels_wsi_np) +', loss_WSI: '+str(wsi_store_loss))
					#print(outputs_wsi_np,labels_np)
					val_file_log.write('{}, {}, {}\n'.format(case_name, pred, labels_wsi_np[0]))
					torch.cuda.empty_cache()
					output_norm = np.where(outputs_wsi_np > 0.5, 1, 0)
					if (TASK=='geneExp'):
						y_pred_val = np.append(y_pred_val, pred)
					else:
						y_pred_val = np.append(y_pred_val,output_norm)
					y_true_val = np.append(y_true_val,labels_np)
					if (TASK=='geneExp'):
						micro_accuracy_valid = MSE(y_true_val, y_pred_val)
					else:
						micro_accuracy_valid = accuracy_micro(y_true_val, y_pred_val)
					print("Val micro_accuracy " + str(micro_accuracy_valid))
				else:
					wsi_store_loss = wsi_store_loss + ((1 / (j+1)) * (loss.item() - wsi_store_loss))
					valid_loss = wsi_store_loss 
					print('output wsi: '+str(outputs_wsi_np)+', label: '+ str(labels_wsi_np) +', loss_WSI: '+str(wsi_store_loss))
					print(outputs_wsi_np,labels_np)
					val_file_log.write('{}, {}, {}\n'.format(case_name, outputs_wsi_np[0], labels_wsi_np[0]))
					torch.cuda.empty_cache()
					output_norm = np.where(outputs_wsi_np > 0.5, 1, 0)
					if (TASK=='geneExp'):
						y_pred_val = np.append(y_pred_val, outputs_wsi_np)    
					else:
						y_pred_val = np.append(y_pred_val,output_norm)
					y_true_val = np.append(y_true_val,labels_np)
					if (TASK=='geneExp'):
						micro_accuracy_valid = MSE(y_true_val, y_pred_val)
					else:
						micro_accuracy_valid = accuracy_micro(y_true_val, y_pred_val)
					print("Val micro_accuracy " + str(micro_accuracy_valid))
					pred=outputs_wsi_np
				if (N_CLASSES==5):
					filenames_wsis = np.append(filenames_wsis,filename_wsi)
					pred_cancers = np.append(pred_cancers,outputs_wsi_np[0])
					pred_hgd = np.append(pred_hgd,outputs_wsi_np[1])
					pred_lgd = np.append(pred_lgd,outputs_wsi_np[2])
					pred_hyper = np.append(pred_hyper,outputs_wsi_np[3])
					pred_normal = np.append(pred_normal,outputs_wsi_np[4])
				else:
					filenames_wsis = np.append(filenames_wsis,filename_wsi)
					pred_cancers = np.append(pred_cancers,pred)

				bool_patches = False

			j = j+1
	
	if (N_CLASSES==5):

		#save_training predictions
		filename_validation_predictions = checkpoint_path+'validation_predictions_'+str(epoch)+'.csv'

		File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers, 'pred_hgd':pred_hgd,'pred_lgd':pred_lgd, 'pred_hyper':pred_hyper,'pred_normal':pred_normal}

		df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper'])
		np.savetxt(filename_validation_predictions, df.values, fmt='%s',delimiter=',')

	else:

		#save_training predictions
		filename_validation_predictions = checkpoint_path+'test_predictions_'+str(epoch)+'.csv'

		File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers}

		df = pd.DataFrame(File,columns=['filenames','pred_cancers'])
		np.savetxt(filename_validation_predictions, df.values, fmt='%s',delimiter=',')
	attention_file.close()
	return valid_loss, wsi_store_loss, patches_store_loss

	#number of epochs without improvement
epoch = 0
if (TASK=='binary' or TASK=='geneExp'):
	iterations = len(train_dataset)
elif (TASK=='multilabel'):
	iterations = len(train_dataset)#+100

tot_batches_training = iterations #int(len(train_dataset)/batch_size_bag)
best_loss = 100000.0

	#number of epochs without improvement
EARLY_STOP_NUM = 12
early_stop_cont = 0
epoch = 0

validation_checkpoints = checkpoint_path+'test_losses/'
create_dir(validation_checkpoints)

NUM_WSI_TO_CLUSTER = 1
THRESHOLD = 0.7
#ALPHA = 1

def entropy_uncertaincy(self,prob):
	i = np.argmax(prob)
	v = entropy(prob, base=2)      
	return v

from sklearn.metrics import mean_squared_error
def MSE(y_true, y_pred):
	#print("[debug] y_true, y_pred: {}, {}".format(y_true, y_pred))
	return mean_squared_error(y_true, y_pred)

if TASK=='geneExp':
	case_files = np.load('data/cms_types_histo_images.npy', allow_pickle=True).item()

pre_train_log_val_acc = open(PRE_TRAIN_LOG_FILE_VAL_ACC, 'w')
pre_train_log_val_acc.write('case name, prediction, label\n')
print("+++++++++++++++++++++ EVAL VALIDATION SPLIT +++++++++++++++++++++++++++++++")
valid_loss, valid_wsi_store_loss, valid_patches_store_loss = evaluate_validation_set(pre_train_log_val_acc, model, -1, validation_generator_bag)    

pre_train_log_val_acc.close()
torch.cuda.empty_cache()
