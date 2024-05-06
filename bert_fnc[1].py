"""
The code uses the BERT model to train a classifier to predict the stance of a given news headline with respect to a given news body. 
The training data is provided in the form of a CSV file containing the news headline, the news body ID, and the correct stance label. 
The script loads the training data, converts the text data into numerical representations suitable for input to the BERT model, 
and trains the model using the AdamW optimizer and a learning rate scheduler. Additionally, the script also evaluates the model on a validation set and 
prints the overall F1 score, as well as the confusion matrix for the predicted labels.
"""



from __future__ import absolute_import, division, print_function

import sys
import argparse
import glob
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import (WEIGHTS_NAME, 
                            BertForSequenceClassification, BertTokenizer,
                            RobertaForSequenceClassification, RobertaTokenizer, 
                            AlbertForSequenceClassification, AlbertTokenizer,
                            get_constant_schedule_with_warmup, 
                            get_linear_schedule_with_warmup,
                            get_cosine_schedule_with_warmup
                        )
from helper import get_labels, get_dev_examples, get_train_examples, get_test_examples, convert_examples_to_features, score_submission, get_f1_overall, print_confusion_matrix


checkpoint_dir = 'checkpoints/'

class hp:
    # Architecture
    model = 'bert'
    model_class = BertForSequenceClassification
    batch_size = 8
    max_len = 512

    # Training
    lr = 3e-05 # This is the learning rate
    lr_type = "linear" # This is the type of learning rate scheduler to use
    num_epochs = 2
    warmup_ratio = 0.06 # The ratio of warmup steps to training steps
    max_norm = 1 # Maximum gradient norm for clipping

    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device to use for training

def load_examples(tokenizer, evaluate=False, train_eval=True):
    label_list = get_labels()
    if train_eval:
        examples = get_dev_examples() if evaluate else get_train_examples()
    else: 
        examples = get_test_examples()
    
    features = convert_examples_to_features(examples, label_list, hp.max_len, tokenizer, 
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(hp.model == 'roberta'),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
        
    input_ids = torch.LongTensor([f.input_ids for f in features])
    input_mask = torch.LongTensor([f.input_mask for f in features])
    segment_ids = torch.LongTensor([f.segment_ids for f in features])
    label_ids = torch.LongTensor([f.label_id for f in features])
    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    return dataset

  #train model using training data 
def train(model, tokenizer):
    # init training structure
    print("training")
    train = load_examples(tokenizer)
    sampler = RandomSampler(train)
    loader = DataLoader(train, sampler=sampler, batch_size=hp.batch_size)
    
    #model.to(hp.device)
    model.zero_grad()
    
    t_total = len(loader) * hp.num_epochs
    warmup_steps = math.ceil(t_total * hp.warmup_ratio)
    
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    
    lr_types = {
        "constant": get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps),
        "linear": get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total),
        "cosine": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    }
    scheduler = lr_types[hp.lr_type]
   
    loss = 0.0
    epoch_num = 0

    """
    The for loop iterates over the training data in batches and passes each batch to the model. It computes the loss function for the predicted labels,
    and updates the model parameters by backpropagating the gradient of the loss function. It clips the gradients to avoid gradient explosion, 
    and updates the learning rate according to the specified schedule.
    
    """
    for _ in range(hp.num_epochs):
        for step, batch in enumerate(loader):
            model.train()
            #batch = tuple(data.to(hp.device) for data in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if hp.model in ['bert', 'albert'] else None, 
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss_func = outputs[0]
                
            loss_func.backward()
            
            #this function clips the gradients of the model parameters to prevent gradients from becoming too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.max_norm)

            loss += loss_func.item()
            
            #weights and learning rate scheduler are updated here 
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        epoch_num += 1
        #save versions of model during certain intervals here 
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        output_dir = os.path.join(checkpoint_dir, 'checkpoint-{}'.format(epoch_num))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)

"""
Evaluates the performance of the model using evaluation data. Calculates evaluation loss and accuracy of the model, F1-score and confusion matrix and 
prints the results
"""
def evaluate(model, tokenizer):
    # init eval structure
    print("evaluating")
    test = load_examples(tokenizer, evaluate=True)
    sampler = SequentialSampler(test)
    loader = DataLoader(test, sampler=sampler, batch_size=hp.batch_size) # Data loader for the evaluation data using the sampler
   
    eval_loss, eval_tr_loss = 0.0, 0.0
    nb_eval_steps = 0
    out_label_ids = None

    checkpoints = [checkpoint_dir]
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(checkpoint_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = hp.model_class.from_pretrained(checkpoint)
        #model.to(hp.device)
        # Loads the model from the checkpoint
        
        # reinitialize preds for every checkpoint
        preds = None
        
        for batch in iter(loader):
            model.eval()
            #batch = tuple(data.to(hp.device) for data in batch)
            
            # No gradient calculation in evaluation mode. Just input data, attention masks, token type IDs 
            with torch.no_grad():
                inputs = {'input_ids':        batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2] if args.model in ['bert', 'albert'] else None, 
                            'labels':         batch[3]}
                outputs = model(**inputs)
                
                # Get loss and logits
                tmp_eval_loss, logits = outputs[:2]

                # for multi gpu, use mean() functionality
                # accumulate loss
                eval_loss += tmp_eval_loss.item()  

            nb_eval_steps += 1

            # calculate average loss within epoch 
            eval_tr_loss = eval_loss / nb_eval_steps
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                
         # convert predictions to class labels                   
        preds = np.argmax(preds, axis=1)

        labels = ['agree', 'disagree', 'discuss', 'unrelated']
        
        acc = accuracy_score(out_label_ids, preds)
        #function calculates the FNC score, which is a metric specific to the Fake News Challenge competition
        fnc_score, conf_matrix = score_submission(preds=preds, labels=out_label_ids)
        fnc_score_best, _ = score_submission(preds=out_label_ids, labels=out_label_ids)
        fnc_score_rel = (fnc_score * 100) / fnc_score_best
        f1, f1_scores = get_f1_overall(labels=labels, conf_matrix=conf_matrix)

        print("\n*******************************************")
        print("EVALUATION OF CHECKPOINT " + checkpoint)
        print_confusion_matrix(conf_matrix)
        print("Score: " + str(fnc_score) + " out of " + str(fnc_score_best) + "\t(" + str(fnc_score*100/fnc_score_best) + "%)")
        print("Accuracy: " + str(acc))
        print("F1 overall: " + str(f1))
        print("F1 per class: " + str(f1_scores))
        print("*******************************************\n")

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', metavar='N',
                        help='Model name. Choices: bert, roberta, albert (default: bert)')
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased', metavar='N',
                        help='Tokenizer name. Choices: bert-base-cased, roberta-base, albert-base-v1 (default: bert-base-cased)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='Number of epochs to train (default: 2)')
    #parser.add_argument('--freeze', type=str, default='freeze_embed', metavar='N',
    #                    help='Freezing technique for finetuning. Choose between "freeze", "no_freeze" and "freeze_embed".')  
    args = parser.parse_args()

    hp.model = args.model
    hp.num_epochs = args.epochs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_types = {
        'bert': (BertForSequenceClassification, BertTokenizer),
        'roberta': (RobertaForSequenceClassification, RobertaTokenizer),
        'albert': (AlbertForSequenceClassification, AlbertTokenizer)
    }

    model_class, tokenizer_class = model_types[args.model]
    hp.model_class = model_class
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer, do_lower_case=False)

    # setting up main directories to save loaded pretrained model and checkpoints
    output_dir_model = os.path.join(args.model, 'model_pretrained')

    # use same randomly initialized classification layers for all experiments
    if os.path.exists(output_dir_model):
        model = model_class.from_pretrained(args.tokenizer, num_labels=4)
    else:
        os.makedirs(output_dir_model)
        model = model_class.from_pretrained(args.tokenizer, num_labels=4)
        model.save_pretrained(output_dir_model)

    train(model, tokenizer)
    evaluate(model, tokenizer)
