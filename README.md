# Linformer Based Conversational  Chatbot

## Contents
- [Introduction](#Introduction)
- [Main requirements](#Main-requirements)
- [Special Thanks](#Special-Thanks)
- [How to Run](#How-to-Run)
- [Folder Structure](#Folder-Structure)
- [Main.py Adjustable Parameters](#Main.py-adjustable-parameters)
- [How to chat with your saved model](#How-to-chat-with-your saved model)
- [Bot.py Adjustable Parameters](#Bot.py-adjustable-parameters)
- [Repository and Code Structure](#Repository-and-code-structure)
- [Results](#Results)
	- [Selecting scheduler and number of epochs](#Selecting-scheduler-and-number-of-epochs)
	- [Where are the correct responses?](#Where-are-the-correct-responses?)
	- [Introduction](#Introduction)
	- [What is the linear dimension for Linformer that reached loss of 2](#What-is-the-linear-dimension-for-Linformer-that-reached-loss-of-2)
	- [Our Linformer Results](#Our-Linformer-Results)
	- [Comparing total execution time at 500 epochs](#Comparing-total-execution-time-at-500-epochs)
- [Conclusion](#Conclusion)
- [References](#References)

## Introduction

Transformer has revolutionized the Natural Language Processing field with the attention mechanism. Some of the groundbreaking NLP models (GPT3 and BERT) of recent years are all based on transformer. However, the time and space complexity of transformer is largely dependent on the size of the input sequence. More specifically, the self-attention mechanism of transformer has a time complexity of O(n2) where n is the length of input sequence. Wang et. al. \[[1](#references)\] proposed Linformer, a linear complexity O(n) self-attention mechanism, that can speed up the inference speed of the model significantly. We sought to find whether Linformer could be used to train and reduce the inference time while still produce a decent outcome in the case of conversational Chatbots, where training input sequences’ lengths are varied but mostly short.

## Main requirements
- Python 3.6 to 3.8 (3.6 preferred)*

\*Require for many dependencies in requirement.txt

## Special Thanks

This GitHub Repository's implementation heavily influenced by Clam004 \[[5](#references)\]

## How to Run
```
# Install dependencies
$ pip install -r requirements.txt

# Test run on default setting
$ python main.py

# A successful run should look like this
```
![main](images/main.jpg)

## Main.py Adjustable Parameters
| Description | DType       | Arguments  | Default | 
| ----------- | ----------- | ---------- | ------- | 
| Name to save weights, at /saved/weights/<name> | string | --weight | "weight" | 
| Name of train data, at /saved/data/<name> | string  | --train | "data2_train_9010" | 
| Name to test data, at /saved/data/<name> | string  | --test | "data2_test_9010" | 
| Batch size | int  | --batch | 32 | 
| # of epochs | int | --epoch | 200 | 
| Modeler: Transformer or Linformer | string | --modeler | "Linformer" | 
| Linear Dimension of Linformer | int | --linear_dimension | 256 | 
| Scheduler: plateau, cosine or warmup | str | --scheduler | "plateau" | 
| Dimension of Attention Layers | int | --dimension | 512 | 
| Number of Attention Layers | int | --nlayers | 6 | 
| Number of Attention Heads | int | --heads | 8 | 
| learning rate | float | --lr | 0.0003 | 

## How to chat with your saved model
```
# For Transformer run
$ python bot.py --model transformer --weight PATH

# For Linformer run
$ python bot.py --model linformer --weight PATH -- linear_dimension SAME_AS_WEIGHT_SETTING

# You should see a prompt like this:
```
![bot](images/bot.jpg)

## Bot.py Adjustable Parameters
| Description | DType       | Arguments  | Default | 
| ----------- | ----------- | ---------- | ------- | 
| Name to save weights, at /saved/weights/<name> | string | --weight | "weight" | 
| Modeler: transformer or linformer | string | --modeler | "linformer" | 
| Linear Dimension of Linformer | int | --linear_dimension | 256 | 

## Repository and Code Structure
- Execution files are locating in the root directory.
- Transfomer model, Linformer model, and Tokenizer script are in "scripts" directory.
- Default saved weight location is "saved/weight" directory.
- Default data location is "saved/data" directory.
      
## Results

### Selecting Learning Rate

- We've found the best learning rate at 0.0003 

<img src="https://github.com/katsamapol/chatbot/blob/main/images/learning_rate.jpg" style="width:50%; margin:auto;">

### Selecting scheduler and number of epochs

- We've chosen "Reduce on Plateau" as our learning rate scheduler.
- Also we've selected 500 epochs for further training because the training losses are flatten after 500 epochs. 

![scheduler](images/scheduler.jpg)

### Where are the correct responses?

- We've seen that at training loss between 1 and 2 the probability of correct response is attractive.

![transformer_table](images/transformer_table.jpg)

### What is the linear dimension for Linformer that reached loss of 2

- K between 32 and 256 could reach loss of 2

![linformer_chart](images/linformer_chart.jpg)

### Our Linformer Results

- K <= 64 works better with simple question
- K >= 128 works better with more sophisticated question

![linformer_table](images/linformer_table.jpg)

### Comparing total execution time at 500 epochs

- Linformer does not reduce total execution time for Conversational Chatbot

<img src="https://github.com/katsamapol/chatbot/blob/main/images/time_to_reach_500_epoch.jpg" style="width:75%; margin:auto;">

## Conclusion

In this project we trained various chatbots with Transformer and Linformer using different learning rates, schedulers, and values for k.

Using a subjective model evaluation method, we determined that lower training loss models can produce higher percentage of valid conversational responses. Linformer models with high k values tended to have similar training loss patterns as regular transformer models. At k<=64 the model worked better with simple utterances while at k>=128 the model seemed to be better at handling more sophisticated utterances. The Linformer was unable to reduce training time or inference time as we had initially hypothesized. Since most conversations contain less than 32 words, Linformer could not utilize E and F matrices effectively. In fact, since the Linformer had 2 extra parameters, in these instances the model took longer to train and test.
	
For further work, the effect of changing the k value to the quality of responses could be studies. During the experimentation, higher k value models tended to handle more sophisticated utterances better.  Furthermore, the study can be repeated with using longer forms of text that have an input length sufficiently larger than then linear dimension. This would enable the Linformer to utilize the E and F matrices.  For instance, the bot could be trained on full movie scripts to generate a movie script of its own or it could be used to generate posts on a forum by looking at other posts.


## <a id="references">References</a>

- [1] Linformer: Self-Attention with Linear Complexity: https://arxiv.org/pdf/2006.04768.pdf
- [2] Cornell Movie-Dialog Corpus Dataset: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
- [3] Deep Learning Based Chatbot Models,  https://arxiv.org/pdf/1908.08835.pdf
- [4] Attention is All You Need, https://arxiv.org/abs/1706.03762
- [5] Clam004 Chat Transformer, https://github.com/clam004/chat-transformer
