# coding=utf-8

"""CUDA_VISIBLE_DEVICES=0 python TAS_BERT_joint.py"""
from __future__ import absolute_import, division, print_function

"""
three-joint detection for target & aspect & sentiment
"""

import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange
import nltk
from gensim import corpora
import tokenization
from modeling import BertConfig, BertForTABSAJoint, BertForTABSAJoint_CRF, BertForTABSAJoint_BiLSTM_CRF
from optimization import BERTAdam
from gensim.models import LdaModel
import datetime
from gensim.test.utils import datapath

from processor import Semeval_Processor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id, ner_label_ids, ner_mask, text_a, text_b):#修改-----------添加a,b句子
		self.input_ids = input_ids#标记编码(batch_size,seq_length)
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.ner_label_ids = ner_label_ids
		self.ner_mask = ner_mask
		self.text_a = text_a#------
		self.text_b = text_b#------


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, ner_label_list, tokenize_method, flag):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i #0/1

	#here start with zero this means that "[PAD]" is zero
	ner_label_map = {}
	for (i, label) in enumerate(ner_label_list):
		ner_label_map[label] = i# [PAD] = 0

	features = []
	all_tokens = []
	#提前准备好所有的句子转换成小写并分词---------------------------
	corpus_a = []
	corpus_b = []
	corpus = []
	filtered_corpus = []
	filtered_corpus_a = []
	filtered_corpus_b = []
	#去停用词
	stop_words = nltk.corpus.stopwords.words('english')
	for w in ['!!!',',','.','?','!','-','cannot',"that's",'...',"don't",'$','/','(',')',"'",":","us"]:
		stop_words.append(w)
	#创建语料库corpus 训练时
	#if flag:
	for (ex_index, example) in enumerate(examples):
		corpus_a.append(nltk.word_tokenize(example.text_a.lower()))
		corpus_b.append(nltk.word_tokenize(example.text_b.lower()))
		corpus.append(nltk.word_tokenize(example.text_a.lower()) + nltk.word_tokenize(example.text_b.lower()))
	for i in range(len(corpus)):
		temp = corpus[i]
		true = [item for item in temp if item not in stop_words]
		filtered_corpus.append(true)
	# for i in range(len(corpus_a)):
	# 	temp = corpus_a[i]
	# 	true = [item for item in temp if item not in stop_words]
	# 	filtered_corpus_a.append(true)
	# for i in range(len(corpus_b)):
	# 	temp = corpus_b[i]
	# 	true = [item for item in temp if item not in stop_words]
	# 	filtered_corpus_b.append(true)
	#filtered_corpus_a.append([item for item in corpus_a if item not in stop_words])
	#filtered_corpus_b.append([item for item in corpus_b if item not in stop_words])
	#创建字典 id2word
	dictionary = corpora.Dictionary(filtered_corpus)
	corpus = [dictionary.doc2bow(s) for s in filtered_corpus]
	#向量化 word2id
	# corpus_a = [dictionary.doc2bow(s) for s in filtered_corpus_a]
	# corpus_b = [dictionary.doc2bow(s) for s in filtered_corpus_b]
	#---------------------------------------------
	for (ex_index, example) in enumerate(tqdm(examples)):
		corpus_a = nltk.word_tokenize(example.text_a.lower())
		corpus_b = nltk.word_tokenize(example.text_b.lower())
		filtered_corpus_a = [item for item in corpus_a if item not in stop_words]
		#print(filtered_corpus_a)
		filtered_corpus_b = [item for item in corpus_b if item not in stop_words]
		#corpus_a = [dictionary.doc2bow(s) for s in filtered_corpus_a]
		#corpus_b = [dictionary.doc2bow(s) for s in filtered_corpus_b]
		if tokenize_method == "word_split":
			# word_split
			word_num = 0
			tokens_a = tokenizer.tokenize(example.text_a)#分词
			ner_labels_org = example.ner_labels_a.strip().split()
			ner_labels_a = []
			token_bias_num = 0

			for (i, token) in enumerate(tokens_a):
				if token.startswith('##'):
					if ner_labels_org[i - 1 - token_bias_num] in ['O', 'T', 'I']:
						ner_labels_a.append(ner_labels_org[i - 1 - token_bias_num])
					else:
						ner_labels_a.append('I')
					token_bias_num += 1
				else:
					word_num += 1
					ner_labels_a.append(ner_labels_org[i - token_bias_num])

			assert word_num == len(ner_labels_org)
			assert len(ner_labels_a) == len(tokens_a)

		else:
			# prefix_match or unk_replace
			tokens_a = tokenizer.tokenize(example.text_a)
			ner_labels_a = example.ner_labels_a.strip().split()

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)

		if tokens_b:
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, ner_labels_a, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]
				ner_labels_a = ner_labels_a[0:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0   0   0   0  0     0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambigiously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = []
		segment_ids = []
		ner_label_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		ner_label_ids.append(ner_label_map["[CLS]"])
		try:
			for (i, token) in enumerate(tokens_a):
				tokens.append(token)
				segment_ids.append(0)
				ner_label_ids.append(ner_label_map[ner_labels_a[i]])
		except:
			print(tokens_a)
			print(ner_labels_a)

		ner_mask = [1] * len(ner_label_ids)
		token_length = len(tokens)
		tokens.append("[SEP]")
		segment_ids.append(0)
		ner_label_ids.append(ner_label_map["[PAD]"])

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
				ner_label_ids.append(ner_label_map["[PAD]"])
			tokens.append("[SEP]")
			segment_ids.append(1)
			ner_label_ids.append(ner_label_map["[PAD]"])

		input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将单词转换成ids

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)# 创建mask
		#创建text_a,text_b语料库-------------------------------------
		#text_a = []
		#text_b = []
		#text_a.append(example.text_a)
		#text_b.append(example.text_b)
		#-----------------------------------------------------------
		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:# 对于输入进行补0
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)
			ner_label_ids.append(ner_label_map["[PAD]"])
		while len(ner_mask) < max_seq_length:
			ner_mask.append(0)
		#---------------------------------------------------------
		#while len(text_a) < max_seq_length:
		#	text_a.append(0)#text的长度应与input_ids一致------------
		#while len(text_b) < max_seq_length:
		#	text_b.append(0)#text的长度应与input_ids一致------------
		#--------------------------------------------------------
		#判断长度---------------------------------------
		#assert len(text_a) == max_seq_length
		#assert len(text_b) == max_seq_length
		#-----------------------------------------------
		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(ner_mask) == max_seq_length
		assert len(ner_label_ids) == max_seq_length

		label_id = label_map[example.label]

		features.append(
				InputFeatures(
						input_ids=input_ids,
						input_mask=input_mask,
						segment_ids=segment_ids,
						label_id=label_id,
						ner_label_ids=ner_label_ids,
						ner_mask=ner_mask,
						text_a=filtered_corpus_a,#添加-------------------------
						text_b=filtered_corpus_b))
		all_tokens.append(tokens[0:token_length])
	return features, all_tokens, corpus, dictionary#--------------

#如果两个Token序列的长度太长，那么需要去掉一些，这会用到_truncate_seq_pair函数：
def _truncate_seq_pair(tokens_a, tokens_b, ner_labels_a, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
			ner_labels_a.pop()
		else:
			tokens_b.pop()

#lda训练函数--------------------------------------------------------------------------
def train_save_lda_model(corpus, id2word):
    '''
    Trains and saves original LDA model from Gensim implementation
    :param corpus: preprocessed corpus
    :param id2word: id2word from preprocessing step
    :param opt: option dictionary
    :return: trained topic model
    '''
    lda_model = LdaModel(corpus=corpus,
                        id2word=id2word,
            	        num_topics=90,
    	            	random_state=100,
                    	update_every=1,
                    	chunksize=100,
                        passes=10,
                	    alpha='auto',
                    	per_word_topics=True)
	#save
    lda_model.save("lda.model")
    return lda_model

def load_lda_model():
	return LdaModel.load("lda.model")

class FreeLB(object):
	def __init__(self, adv_K=3, adv_lr=2e-2, adv_init_mag=1e-2, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
		self.adv_K = adv_K
		self.adv_lr = adv_lr
		self.adv_max_norm = adv_max_norm
		self.adv_init_mag = adv_init_mag
		self.adv_norm_type = adv_norm_type
		self.base_model = base_model

	def attack(self, model, inputs, segment_ids, label_ids, ner_label_ids, ner_mask, lda_model, corpus_a, corpus_b, gradient_accumulation_steps=1):
		input_ids = inputs['input_ids']
		if isinstance(model, torch.nn.DataParallel):
			embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
		else:
			embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
		if self.adv_init_mag > 0:
			input_mask = inputs['attention_mask'].to(embeds_init)
			input_lengths = torch.sum(input_mask, 1)
			if self.adv_norm_type == "l2":
				delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
				dims = input_lengths * embeds_init.size(-1)
				mag = self.adv_init_mag / torch.sqrt(dims)
				delta = (delta * mag.view(-1, 1, 1)).detach()
			elif self.adv_norm_type == "linf":
				delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
				delta = delta * input_mask.unsqueeze(2)
		else:
			delta = torch.zeros_like(embeds_init)

		for astep in range(self.adv_K):
			delta.requires_grad_()
			inputs['inputs_embeds'] = delta + embeds_init
			inputs['input_ids'] = None
			loss, ner_loss, _, _ = model(inputs['input_ids'], segment_ids, inputs['attention_mask'], label_ids, ner_label_ids, ner_mask, False, lda_model, corpus_a, corpus_b, inputs['inputs_embeds'])
			#loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
			loss = loss.mean()  # mean() to average on multi-gpu parallel training
			ner_loss = ner_loss.mean()
			loss = loss / gradient_accumulation_steps
			ner_loss = ner_loss / gradient_accumulation_steps
			loss.backward(retain_graph=True)
			ner_loss.backward()
			delta_grad = delta.grad.clone().detach()
			if self.adv_norm_type == "l2":
				denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
				denorm = torch.clamp(denorm, min=1e-8)
				delta = (delta + self.adv_lr * delta_grad / denorm).detach()
				if self.adv_max_norm > 0:
					delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
					exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
					reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
					delta = (delta * reweights).detach()
			elif self.adv_norm_type == "linf":
				denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
				denorm = torch.clamp(denorm, min=1e-8)
				delta = (delta + self.adv_lr * delta_grad / denorm).detach()
				if self.adv_max_norm > 0:
					delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
			else:
				raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
			if isinstance(model, torch.nn.DataParallel):
				embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
			else:
				embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
		return loss, ner_loss

#---------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default='data/semeval2015/three_joint/TO/',
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--output_dir",
						default='results/semeval2015/three_joint/TO/my_result',
						type=str,
						required=True,
						help="The output directory where the model checkpoints will be written.")
	parser.add_argument("--vocab_file",
						default='uncased_L-12_H-768_A-12/vocab.txt',
						type=str,
						required=True,
						help="The vocabulary file that the BERT model was trained on.")
	parser.add_argument("--bert_config_file",
						default='uncased_L-12_H-768_A-12/bert_config.json',
						type=str,
						required=True,
						help="The config json file corresponding to the pre-trained BERT model. \n"
							 "This specifies the model architecture.")
	parser.add_argument("--init_checkpoint",
						default='uncased_L-12_H-768_A-12/pytorch_model.bin',
						type=str,
						required=True,
						help="Initial checkpoint (usually from a pre-trained BERT model).")
	parser.add_argument("--tokenize_method",
						default='word_split',
						type=str,
						required=True,
						choices=["prefix_match", "unk_replace", "word_split"],
						help="how to solve the unknow words, max prefix match or replace with [UNK] or split to some words")
	parser.add_argument("--use_crf",
						default=True,
						required=True,
						action='store_true',
						help="Whether to use CRF after Bert sequence_output")

	## Other parameters
	parser.add_argument("--eval_test",
						default=True,
						action='store_true',
						help="Whether to run eval on the test set.")
	parser.add_argument("--do_lower_case",
						default=True,
						action='store_true',
						help="Whether to lower case the input text. True for uncased models, False for cased models.")
	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--train_batch_size",
						default=24,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=2e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=30.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						default=False,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--accumulate_gradients",
						type=int,
						default=1,
						help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=53,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumualte before performing a backward/update pass.")
	args = parser.parse_args()


	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

	if args.accumulate_gradients < 1:
		raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
							args.accumulate_gradients))

	args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)
	# 随机数种子
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	bert_config = BertConfig.from_json_file(args.bert_config_file)

	if args.max_seq_length > bert_config.max_position_embeddings:
		raise ValueError(
			"Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
			args.max_seq_length, bert_config.max_position_embeddings))

	processor = Semeval_Processor()
	label_list = processor.get_labels()
	ner_label_list = processor.get_ner_labels(args.data_dir)    # BIO or TO tags for ner entity ['[PAD]', '[CLS]', 'O', 'B', 'I']

	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, tokenize_method=args.tokenize_method, do_lower_case=args.do_lower_case)

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
		raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
	os.makedirs(args.output_dir, exist_ok=True)

	# training set
	train_examples = None
	num_train_steps = None
	train_examples = processor.get_train_examples(args.data_dir)# 获取训练数据
	num_train_steps = int(
		len(train_examples) / args.train_batch_size * args.num_train_epochs)
	# 传入训练数据
	flag = True
	print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	train_features, _ , corpus, dictionary= convert_examples_to_features(
		train_examples, label_list, args.max_seq_length, tokenizer, ner_label_list, args.tokenize_method, flag)
	print("sssssssssssssssssssssssssssssssssssssssss")
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_examples))
	logger.info("  Batch size = %d", args.train_batch_size)
	logger.info("  Num steps = %d", num_train_steps)
	print(len(ner_label_list))
	all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
	all_ner_label_ids = torch.tensor([f.ner_label_ids for f in train_features], dtype=torch.long)
	all_ner_mask = torch.tensor([f.ner_mask for f in train_features], dtype=torch.long)# mask掩码
	#-------------------------------------
	all_text_a = []
	all_text_b = []
	all_text_a = [f.text_a for f in train_features]
	all_text_b = [f.text_b for f in train_features]
	all_corpus_a = [dictionary.doc2bow(s) for s in all_text_a]
	#print(len(all_corpus_a))
	all_text_a = []
	for i in range(len(all_corpus_a)):
		while len(all_corpus_a[i]) < 128:
			temp = all_corpus_a[i]
			temp.append((0,0))
		all_text_a.append(temp)
	#print(all_text_a)
	print(len(all_text_a))
	all_corpus_a = torch.tensor([item for item in all_text_a])
	all_corpus_b = torch.tensor([dictionary.doc2bow(s) for s in all_text_b])
	#-------------------------------------
	# 使用TensorDataset生成数据集
	train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ner_label_ids, all_ner_mask, all_corpus_a, all_corpus_b)#修改-----
	# 修改--------------------------------------------------------------------------------
	if args.local_rank == -1:# 采样方式
		#train_sampler = WeightedRandomSampler([1,38],len(train_examples))
		train_sampler = RandomSampler(train_data)
	else:
		train_sampler = DistributedSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
	# ---------------------------------------------------------------------------------------


	# test set
	if args.eval_test:
		flag = False
		test_examples = processor.get_test_examples(args.data_dir)# 获取样本
		test_features, test_tokens, _, _ = convert_examples_to_features(# 将样本转变为特征
			test_examples, label_list, args.max_seq_length, tokenizer, ner_label_list, args.tokenize_method, flag)

		all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
		all_ner_label_ids = torch.tensor([f.ner_label_ids for f in test_features], dtype=torch.long)
		all_ner_mask = torch.tensor([f.ner_mask for f in test_features], dtype=torch.long)
		#-------------------------------------
		all_text_a = []
		all_text_b = []
		all_text_a = [f.text_a for f in test_features]
		all_text_b = [f.text_b for f in test_features]
		all_corpus_a = [dictionary.doc2bow(s) for s in all_text_a]
		#print(len(all_corpus_a))
		all_text_a = []
		for i in range(len(all_corpus_a)):
			while len(all_corpus_a[i]) < 128:
				temp = all_corpus_a[i]
				temp.append((0,0))
			all_text_a.append(temp)
		#print(all_text_a)
		#print(len(all_text_a))
		all_corpus_a = torch.tensor([item for item in all_text_a])
		#print(all_corpus_a)
		all_corpus_b = torch.tensor([dictionary.doc2bow(s) for s in all_text_b])
		#-------------------------------------

		test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ner_label_ids, all_ner_mask, all_corpus_a, all_corpus_b)#添加
		test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)


	# model and optimizer
	if args.use_crf:
		model = BertForTABSAJoint_CRF(bert_config, len(label_list), len(ner_label_list))
		#model = BertForTABSAJoint_BiLSTM_CRF(bert_config, len(label_list), len(ner_label_list))
	else:
		model = BertForTABSAJoint(bert_config, len(label_list), len(ner_label_list), args.max_seq_length)

	if args.init_checkpoint is not None:
		model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
	model.to(device)

	if args.local_rank != -1:# 并行gpu计算
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)
	# 权重衰减
	no_decay = ['bias', 'gamma', 'beta']
	optimizer_parameters = [
		 {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
		 {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
		 ]

	optimizer = BERTAdam(optimizer_parameters,
						 lr=args.learning_rate,
						 warmup=args.warmup_proportion,
						 t_total=num_train_steps)

	print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
	

	# train
	output_log_file = os.path.join(args.output_dir, "log.txt")
	print("output_log_file=",output_log_file)
	with open(output_log_file, "w") as writer:
		if args.eval_test:
			writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
		else:
			writer.write("epoch\tglobal_step\tloss\n")

	global_step = 0
	epoch=0
	# 训练总的lda-------------------------------------------------------------
	lda_model = load_lda_model()
	#lda_model = train_save_lda_model(corpus, dictionary)
	print("-------主题模型训练完成！--------")
	#-----------------------------------------------------------------------
	for _ in trange(int(args.num_train_epochs), desc="Epoch"):
		epoch+=1
		model.train()
		tr_loss = 0
		tr_ner_loss = 0
		nb_tr_examples, nb_tr_steps = 0, 0
		for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
			batch = tuple(t.to(device) for t in batch)# 取出每一batch的编码数据
			input_ids, input_mask, segment_ids, label_ids, ner_label_ids, ner_mask, corpus_a, corpus_b = batch# 修改
			#----------------------------------------------------
			inputs = {"input_ids":input_ids, "attention_mask":input_mask, "inputs_embeds":None}
			freelb = FreeLB()

			#corpus_a = [dictionary.doc2bow(s) for s in corpus_a]
			#corpus_b = [dictionary.doc2bow(s) for s in corpus_b]
			
			corpus_a = corpus_a.cpu().numpy().tolist()
			corpus_b = corpus_b.cpu().numpy().tolist()
			true_a = []
			true_b = []
			for i in range(len(corpus_a)):
				temp = []
				for j in range(len(corpus_a[i])):
					if corpus_a[i][j] != [0,0]:
						temp.append(tuple(corpus_a[i][j]))
				true_a.append(temp)
			#print(true_a)
			# for i in range(len(corpus_b)):
			# 	temp = []
			# 	for j in range(len(corpus_b[i])):
			# 		if corpus_a[i][j] != [0,0]:
			# 			temp.append(tuple(corpus_b[i][j]))
			# 	true_b.append(temp)
			corpus_a = true_a
			cprpus_b = true_b
			#----------------------------------------------------
			loss, ner_loss = freelb.attack(model, inputs, segment_ids, label_ids, ner_label_ids, ner_mask, lda_model, corpus_a, corpus_b, gradient_accumulation_steps=1)
			#loss, ner_loss, _, _ = model(input_ids, segment_ids, input_mask, label_ids, ner_label_ids, ner_mask, False, lda_model, corpus_a, corpus_b)# 修改 训练时使用logit adjustment

			if n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu.
				ner_loss = ner_loss.mean()
			if args.gradient_accumulation_steps > 1:# 解决显存问题
				loss = loss / args.gradient_accumulation_steps
				ner_loss = ner_loss / args.gradient_accumulation_steps
			#loss.backward(retain_graph=True)# 保留backward后的中间参数
			#ner_loss.backward()

			tr_loss += loss.item()# 固定形式
			tr_ner_loss += ner_loss.item()
			nb_tr_examples += input_ids.size(0)
			nb_tr_steps += 1
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()    # We have accumulated enought gradients
				model.zero_grad()
				global_step += 1


		# eval_test
		if args.eval_test:

			model.eval()
			test_loss, test_accuracy = 0, 0
			ner_test_loss = 0
			nb_test_steps, nb_test_examples = 0, 0
			with open(os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt"),"w") as f_test:
				f_test.write('yes_not\tyes_not_pre\tsentence\ttrue_ner\tpredict_ner\n')
				batch_index = 0
				for input_ids, input_mask, segment_ids, label_ids, ner_label_ids, ner_mask, corpus_a, corpus_b in test_dataloader:# 记载测试集
					input_ids = input_ids.to(device)
					input_mask = input_mask.to(device)
					segment_ids = segment_ids.to(device)
					label_ids = label_ids.to(device)
					ner_label_ids = ner_label_ids.to(device)
					ner_mask = ner_mask.to(device)
					#-----------------------------------------------------
					#corpus_a = [dictionary.doc2bow(s) for s in corpus_a]
					#corpus_b = [dictionary.doc2bow(s) for s in corpus_b]
					#corpus_a = corpus_a.to(device)
					#corpus_b = corpus_b.to(device)

					corpus_a = corpus_a.cpu().numpy().tolist()
					corpus_b = corpus_b.cpu().numpy().tolist()
					true_a = []
					true_b = []
					for i in range(len(corpus_a)):
						temp = []
						for j in range(len(corpus_a[i])):
							if corpus_a[i][j] != [0,0]:
								temp.append(tuple(corpus_a[i][j]))
						true_a.append(temp)
					#print(true_a)
					# for i in range(len(corpus_b)):
					# 	temp = []
					# 	for j in range(len(corpus_b[i])):
					# 		if corpus_a[i][j] != [0,0]:
					# 			temp.append(tuple(corpus_b[i][j]))
					# 	true_b.append(temp)
					corpus_a = true_a
					cprpus_b = true_b
					#-----------------------------------------------------
					# test_tokens is the origin word in sentences [batch_size, sequence_length]
					ner_test_tokens = test_tokens[batch_index*args.eval_batch_size:(batch_index+1)*args.eval_batch_size]
					batch_index += 1

					with torch.no_grad():# 这里logits是经过BERT+双层Linear得到的未softmax的概率
						tmp_test_loss, tmp_ner_test_loss, logits, ner_predict = model(input_ids, segment_ids, input_mask, label_ids, ner_label_ids, ner_mask, args.eval_test, lda_model, corpus_a, corpus_b)# eval_test进行控制

					# category & polarity
					logits = F.softmax(logits, dim=-1)# softmax归一化后概率分布
					logits = logits.detach().cpu().numpy()
					label_ids = label_ids.to('cpu').numpy()
					outputs = np.argmax(logits, axis=1)# 选择yes/no

					if args.use_crf:
						# CRF
						ner_logits = ner_predict# CRF后的最大概率的标签序列
					else:
						# softmax
						ner_logits = torch.argmax(F.log_softmax(ner_predict, dim=2),dim=2)# 标签的softmax概率
						ner_logits = ner_logits.detach().cpu().numpy()

					ner_label_ids = ner_label_ids.to('cpu').numpy()
					ner_mask = ner_mask.to('cpu').numpy()


					for output_i in range(len(outputs)):
						# category & polarity
						f_test.write(str(label_ids[output_i]))
						f_test.write('\t')
						f_test.write(str(outputs[output_i]))
						f_test.write('\t')

						# sentence & ner labels
						sentence_clean = []
						label_true = []
						label_pre = []
						sentence_len = len(ner_test_tokens[output_i])

						for i in range(sentence_len):
							if not ner_test_tokens[output_i][i].startswith('##'):
								sentence_clean.append(ner_test_tokens[output_i][i])
								label_true.append(ner_label_list[ner_label_ids[output_i][i]])
								label_pre.append(ner_label_list[ner_logits[output_i][i]])

						f_test.write(' '.join(sentence_clean))# 用' '来区分开join中的列表
						f_test.write('\t')
						f_test.write(' '.join(label_true))
						f_test.write("\t")
						f_test.write(' '.join(label_pre))
						f_test.write("\n")
					tmp_test_accuracy=np.sum(outputs == label_ids)
					test_loss += tmp_test_loss.mean().item()
					ner_test_loss += tmp_ner_test_loss.mean().item()
					test_accuracy += tmp_test_accuracy

					nb_test_examples += input_ids.size(0)
					nb_test_steps += 1

			test_loss = test_loss / nb_test_steps
			ner_test_loss = ner_test_loss / nb_test_steps
			test_accuracy = test_accuracy / nb_test_examples


		result = collections.OrderedDict()
		if args.eval_test:
			result = {'epoch': epoch,
					'global_step': global_step,
					'loss': tr_loss/nb_tr_steps,
					'test_loss': test_loss,
					'ner_test_loss': ner_test_loss,
					'test_accuracy': test_accuracy}
		else:
			result = {'epoch': epoch,
					'global_step': global_step,
					'loss': tr_loss/nb_tr_steps,
					'ner_loss': tr_ner_loss / nb_tr_steps}

		logger.info("***** Eval results *****")
		with open(output_log_file, "a+") as writer:
			for key in result.keys():
				logger.info("  %s = %s\n", key, str(result[key]))
				writer.write("%s\t" % (str(result[key])))
			writer.write("\n")

if __name__ == "__main__":
	main()
