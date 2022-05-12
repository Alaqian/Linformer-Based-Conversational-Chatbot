import math, copy, sys
import torch
import argparse

from scripts.MoveData import *
from scripts.Linformer import *
from scripts.Transformer import *
from scripts.TalkTrain import *

def main():
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	print(device)
	#Parse command line args
	parser = argparse.ArgumentParser(description='Create Chatbot Instance')
	
	parser.add_argument('-w', '--weight', default="data_weight", type=str, help='Name to save weights at /saved/weights/<name>')
	parser.add_argument('-m', '--model', default="transformer", type=str, help='Model: Transformer or Linformer')

	args = parser.parse_args()
	
	opt = Options(batchsize=32, device=torch.device("cpu"), epochs=200, lr=0.01, max_len = 32, save_path = f'saved/weights/{args.weight}')
	data_iter, infield, outfield, opt = json2datatools(path = f'saved/data/data2_train_9010.json', opt=opt)
	emb_dim, n_layers, heads, dropout = 512, 6, 8, 0.1 

	if(args.model.lower() == "transformer"):
		model = Transformer(len(infield.vocab), len(outfield.vocab), emb_dim, n_layers, heads, dropout)
	elif(args.model.lower() == "linformer"):
		model = Linformer(len(infield.vocab), len(outfield.vocab), emb_dim, opt.max_len, n_layers, heads, dropout)
	else:
		print("Please choose a model between \"Transformer\" and \"Linformer\"")
		quit()
	
	model.load_state_dict(torch.load(opt.save_path, map_location=torch.device('cpu'))) 
	while True:
		your_input = input("You > ")
		if(not your_input.endswith(".") and not your_input.endswith("!") and not your_input.endswith("?")):
			model_input = your_input + "."
		else:
			model_input = your_input
		bot_reply = talk(model_input, model, opt, infield, outfield)
		if ("bye" in your_input or "bye ttyl" in bot_reply):
			print('Bot > '+ bot_reply + '\n')
			break
		else:
			print('Bot > '+ bot_reply + '\n') 
if __name__ == "__main__":
	main()