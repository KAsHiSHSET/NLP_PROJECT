import string
from unittest import result
from django.shortcuts import render
from django.http import HttpResponse
import shutil

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
from tokenizers import BertWordPieceTokenizer
import numpy as np

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
bert_layer = hub.KerasLayer(tfhub_handle_encoder,trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
tokenizer = BertWordPieceTokenizer(vocab=vocab_file)

class Sample:
    def __init__(self, question, context, start_char_idx=None, answer_text=None, answer_exist=1):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.answer_exist = answer_exist
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self):
        # tokenize context and question
        tokenized_context = tokenizer.encode(self.context)
        tokenized_question = tokenizer.encode(self.question)
        
        # if this is validation or training sample, preprocess answer
        if self.answer_text is not None:
            # check if end character index is in the context
            end_char_idx = self.start_char_idx + len(self.answer_text)
            if end_char_idx >= len(self.context):
                self.skip = True
                return
        
            # mark all the character indexes in context that are also in answer     
            is_char_in_ans = [0] * len(self.context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1
            ans_token_idx = []
        
            # find all the tokens that are in the answers
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)
        
            if len(ans_token_idx) == 0:
                self.skip = True
                return
        
            # get start and end token indexes
            if self.answer_exist == 1:
                self.start_token_idx = ans_token_idx[0]
                self.end_token_idx = ans_token_idx[-1]
            else:
                self.start_token_idx = 0
                self.end_token_idx = 0
                
        # create inputs as usual
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        
        # add padding if necessary
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        
        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
model_name = '../bert-chaii'
max_seq_length = 512

word_sep = ' '
max_word_len = 128
word_overlap = 64

def create_bert_inputs(samples):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }

    for item in samples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_word_ids"],
        dataset_dict["input_mask"],
        dataset_dict["input_type_ids"]]

    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    
    return x, y

def make_spans(question, context, answer_start=None, answer_text=None):
    span_texts, words = make_span_texts(context)
    spans = []
    total_char_len = 0
    span_no = 0

    for s in span_texts:
        span_context = word_sep.join(s)
        span_char_len = len(span_context)
        span_answer_start = 0
        span_answer_exist = 0
        
        if answer_text is not None:
            if (answer_start >= total_char_len and  answer_start < (total_char_len + span_char_len)): # Answer exist
                span_answer_exist = 1
                if span_no > 0:
                    span_answer_start = answer_start - total_char_len
                else:
                    span_answer_start =  answer_start # for span 0
            
        span = [question, span_context, span_answer_start, answer_text, span_answer_exist]
        spans.append(span)
        
        total_char_len = total_char_len + span_char_len
        span_no = span_no + 1

    return spans

def make_span_texts(text):
    words = text.split(word_sep)
    nwords = len(words)
    remainder = nwords % max_word_len
    nspans = int((nwords - remainder) / max_word_len) + 1
    spans = []

    for x in range(nspans):
        start = x*(max_word_len-word_overlap)
        end = start + max_word_len
        if(end > nwords): end = nwords
        span = words[start:end]
        spans.append(span)

    return spans, words

def chaii_test_data(ques, text):
    samples = []
    question = ques
    context = text
    spans = make_spans(question, context)
    for s in spans:
        s = Sample(s[0], s[1])
        s.preprocess()
        samples.append(s)
            
    return samples

model = tf.keras.models.load_model(model_name)

def testModel(ques,text):
    
    #model = tf.keras.models.load_model(model_name)
    test_samples = chaii_test_data(ques,text)
    test_samples = [x for x in test_samples if x.skip == False]
    xt, _ = create_bert_inputs(test_samples)
    
    pred_start, pred_end = model.predict(xt)
    
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        
        test_sample = test_samples[idx]
        offsets = test_sample.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        pred_ans = None
        
        if start >= end : continue
        if start >= len(offsets): continue

        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
        else:
            pred_ans = test_sample.context[pred_char_start:]
        
        print("Q: " + test_sample.question)
        print("A: " + pred_ans)
        return pred_ans

#testModel("लूथर के द्वारा लिखा गया बाद में क्या खोजा गया?","एक कागज का टुकड़ा बाद में पाया गया जिसपर लूथर ने अपनाआखिरी कथन लिखा। यह कथन “हम याचक हैं” जो जर्मन में था,  के अतिरिक्त लैटिन में था।")

def Welcome(request): 
    return render(request, 'welcome.html')

def Index(request): 
    # passage=request.GET['passage']
    # ques=request.GET['question']
    # ans=testModel(ques,passage)
    return render(request, 'index.html')
    # else:
    #     return render(request,'index.html',{'result':ans,'question':ques,'passage':passage})

    

def Result(request):
    passage=request.GET['passage']
    ques=request.GET['question']
    ans=testModel(ques,passage)
    return render(request, 'result.html',{'result':ans,'question':ques,'passage':passage})
