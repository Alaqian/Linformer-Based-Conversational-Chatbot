import math, copy, sys
import torch
import argparse
from linformer_pytorch import LinformerEncDec

from scripts.MoveData import *
from scripts.Transformer import *
from scripts.Linformer import *
from scripts.TalkTrain import *

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# print(device)
	# Parse command line args
	parser = argparse.ArgumentParser(description='Transformer&Linformer chatbot trainer')

	parser.add_argument('-w', '--weight', default="weight", type=str, help='Name to save weights at /saved/weights/<name>')
	parser.add_argument('-tr', '--train', default="data2_train_9010", type=str, help='Name to train file at /saved/data/<name>')
	parser.add_argument('-te', '--test', default="data2_test_9010", type=str, help='Name to test file at /saved/data/<name>')
	parser.add_argument('--batch', default=32, type=int, help='Batch size')
	parser.add_argument('--epoch', default=200, type=int, help='# of epochs')
	parser.add_argument('--shuffle', default=False, help='Shuffle validation data?')
	parser.add_argument('--verbose', default=False, help='Print adaptive learning rate?')
	parser.add_argument('--modeler', default="linformer", help='Type: transformer or linformer')
	parser.add_argument('--scheduler', default="plateau", help='Scheduler: plateau, cosine or warmup')
	parser.add_argument('--linear_dimension', default=256, type=int, help='Linear Dimension of Attention Layers')
	parser.add_argument('--dimension', default=512, type=int, help='Dimension of Attention Layers')
	parser.add_argument('--nlayers', default=6, type=int, help='Number of Attention Layers')
	parser.add_argument('--heads', default=8, type=int, help='Number of Attention Heads')
	parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
	args = parser.parse_args()

	print('==> Program Start..')
	print(f'==> Batch Size: {args.batch}')
	print(f'==> Number of epochs: {args.epoch}')
	print(f'==> Shuffle: {args.shuffle}')
	print(f'==> Verbose: {args.verbose}')
	if args.verbose == "True":
		args.verbose = True
	print(f'==> Modeler: {args.modeler}')
	if args.modeler.lower() == "linformer":
		print(f'==> Linear Dimension: {args.linear_dimension}')
	print(f'==> Learning rate: {args.lr}')
	print(f'==> Scheduler: {args.scheduler}')
	print(f'==> Name to save weights at saved/weights/{args.weight}')
	print(f'==> Name to train data at saved/data/{args.train}')
	print(f'==> Name to test data at saved/data/{args.test}')
	
	opt = Options(batchsize=args.batch, device=torch.device(device), epochs=args.epoch, lr=args.lr, max_len = 25, save_path = f'saved/weights/{args.weight}')
	print('==> Load Dataset..')
	train_data_iter, train_infield, train_outfield, train_opt = json2datatools(path = f'saved/data/{args.train}.json', opt=opt, train=True, shuffle=True)
	print('train input vocab size', len(train_infield.vocab), 'train reply vocab size', len(train_outfield.vocab))
	test_data_iter, test_infield, test_outfield, test_opt = json2datatools(path = f'saved/data/{args.test}.json', opt=opt, train=False, shuffle=args.shuffle)
	print('test input vocab size', len(test_infield.vocab), 'test reply vocab size', len(test_outfield.vocab))
	print("==> Number of train steps per epoch",num_batches(train_data_iter))
	print("==> Number of test steps per epoch",num_batches(train_data_iter))
	print('==> Build Model..' )

	# Attention is All You Need's setting
	emb_dim, n_layers, heads, dropout = args.dimension, args.nlayers, args.heads, 0.1
	# Linear Attenion's setting
	linear_dimension = args.linear_dimension

	#There is no theoretical limit on the input length (ie number of tokens for a sentence in NLP) 
	#However in practice, longer inputs will consume more memory.
	if(args.modeler=="transformer"):
		model = Transformer(
			len(train_infield.vocab), 
			len(train_outfield.vocab), 
			emb_dim, 
			n_layers, 
			heads, 
			dropout
		)

	elif(args.modeler=="linformer"):
		model = Linformer(
			len(train_infield.vocab), 
			len(train_outfield.vocab), 
			emb_dim, 
			linear_dimension,
			n_layers, 
			heads, 
			dropout
		)
	else:
		print("Please choose modeler between \"transformer\" and \"linformer\"")
		quit()

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
	#warmup_steps = 4000
	if args.scheduler == "warmup":
		scheduler = AdamWarmup(model_size = emb_dim, warmup_steps = 4000, optimizer = optimizer, verbose=args.verbose)
		scheduler.print_lr()
		# scheduler = AdamWarmup(model_size = emb_dim, warmup_steps = num_batches(train_data_iter)*(args.epoch*0.1), optimizer = optimizer, verbose=args.verbose)
	elif args.scheduler == "cosine":
		# scheduler = CosineWithRestarts(optimizer, T_max=num_batches(train_data_iter), verbose=args.verbose)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, num_batches(train_data_iter), T_mult=1, eta_min=0, last_epoch=-1, verbose=args.verbose)
	elif args.scheduler == "plateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=3, verbose=args.verbose)
	else:
		print("Please choose a scheduler between \"cosine (with restart)\" and \"warmup\"")
		quit()

	print('==> Start Training..' , flush=True)
	trainer(model, train_data_iter, train_opt, test_data_iter, test_opt, optimizer, scheduler, args.scheduler)

if __name__ == "__main__":
	main()	