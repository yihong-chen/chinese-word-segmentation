import os
import torch
from torch import nn, optim, autograd
from sklearn.cross_validation import train_test_split

from util.lstm_tagger import LSTMTagger
from util.tuning import train_dev_split, report_f_measure, use_cuda
from util.annotator import annotate, annotate_list_of_sents, segment


def create_dictionary(sequences, start_from_zero=True):
    """Create dictionary for elements in the sequences
    Note that the 0 index is not used

    args:
        sequences: list of list of elements
    """
    sequences = list(set([element for seq in sequences for element in seq]))
    if start_from_zero:
        return {k:v for k,v in zip(sequences, list(range(0, len(sequences))))}
    else:
        return {k: v for k, v in zip(sequences, list(range(1, len(sequences) + 1)))}


def prepare_sequence(sequences, to_idx, out_of_dict_index=0):
    """Convert element sequences into indices sequence"""
    idx_seqs = []
    for seq in sequences:
        idx_seq = [to_idx[element] if element in to_idx else out_of_dict_index for element in seq]
        idx_seqs.append(idx_seq)
    return idx_seqs


def test(lstm_params,
         model, loss_func,
         test_sentences_idx, test_labels_idx,
         idx2label, test_sentences,
         train_dict_file, test_file_gold, cache_dir='./output/'):
    model.eval()
    test_preds = []
    avg_loss = 0
    for sen, target in zip(test_sentences_idx, test_labels_idx):
        sen = autograd.Variable(torch.LongTensor(sen))
        target = autograd.Variable(torch.LongTensor(target))
        if lstm_params['use_cuda']:
            sen = sen.cuda()
            target = target.cuda()
        tag_scores = model(sen)
        loss = loss_func(tag_scores, target)
        if lstm_params['use_cuda']:
            avg_loss += loss.cpu().data.numpy()[0]
        else:
            avg_loss += loss.data.numpy()[0]

        tag = torch.max(tag_scores, dim=1)[1]
        test_preds.append(tag.data.tolist())
    avg_loss = avg_loss / (len(test_sentences_idx) * 1.0)

    # convert label into element
    test_preds = prepare_sequence(test_preds, idx2label)
    test_tagged = os.path.join(cache_dir, 'lstm_test_info.txt')
    annotate_list_of_sents(test_sentences, test_preds, output_file=test_tagged)

    # segment the text according to the label
    segmented_test = os.path.join(cache_dir, 'lstm_test_segmented.txt')
    if os.path.exists(segmented_test):
        os.remove(segmented_test) # clear model cache
    segment(test_tagged, segmented_test)

    # score the result
    score_info = os.path.join(cache_dir, 'lstm_score_info.txt')
    if os.path.exists(score_info):
        os.remove(score_info) # clear model cache
    score_cmd = 'perl score {} {} {} > {}'.format(train_dict_file,
                                                  test_file_gold,
                                                  segmented_test,
                                                  score_info)
    os.system(score_cmd)
    f_measure = report_f_measure(score_info)
    return avg_loss, f_measure


def tune_lstm_hyper(lstm_params):

    # train_dev_split('./data/pku_training.utf8',
    #                 './data/pku_tune_training.utf8',
    #                 './data/pku_tune_training_words.utf8',
    #                 './data/pku_tune_dev.utf8',
    #                 './data/pku_tune_dev_gold.utf8')

    # train_dict_file = './data/pku_tune_training_words.utf8'
    # test_file_gold = './data/pku_tune_dev_gold.utf8'

    train_dict_file = './data/pku_training_words.utf8'
    test_file_gold = './data/pku_test_gold.utf8'

    train_sentences, train_labels = annotate('./data/pku_training.utf8', return_sequence=True)
    test_sentences, test_labels = annotate('./data/pku_test.utf8', return_sequence=True)

    # construt dictionary to map (word, idx) or (label, idx)
    word2idx = create_dictionary(train_sentences, start_from_zero=False)
    label2idx = create_dictionary(train_labels, start_from_zero=True)
    idx2word = {v:k for k,v in word2idx.items()}
    idx2label = {v:k for k,v in label2idx.items()}

    # prepare sequence
    train_sentences_idx = prepare_sequence(train_sentences, word2idx)
    train_labels_idx = prepare_sequence(train_labels, label2idx)
    test_sentences_idx = prepare_sequence(test_sentences, word2idx)
    test_labels_idx = prepare_sequence(test_labels, label2idx, out_of_dict_index=0)

    # define model
    model = LSTMTagger(lstm_params['embedding_dim'],
                       lstm_params['hidden_dim'],
                       len(word2idx) + 1,
                       len(label2idx),
                       lstm_params['use_cuda'])
    if lstm_params['use_cuda']:
        use_cuda(enabled=True, device_id=lstm_params['device_id'])
        model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(300):
        model.train()
        avg_loss = 0
        for sentence, tags in zip(train_sentences_idx, train_labels_idx):
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = autograd.Variable(torch.LongTensor(sentence))
            targets = autograd.Variable(torch.LongTensor(tags))
            if lstm_params['use_cuda']:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            if lstm_params['use_cuda']:
                avg_loss += loss.cpu().data.numpy()[0]
            else:
                avg_loss += loss.data.numpy()[0]
        avg_loss = avg_loss / (len(train_sentences_idx) * 1.0)
        print('[Training] Epoch {}, Average Loss {}'.format(epoch, avg_loss))

        # Evaluate every epoch
        test_loss, test_f1 = test(lstm_params,
                                  model, loss_function,
                                  test_sentences_idx, test_labels_idx,
                                  idx2label, test_sentences,
                                  train_dict_file, test_file_gold, cache_dir='./output/')
        print('[Evaluate] Epoch {}, Average Loss {}, F1 measure {}'.format(epoch, test_loss, test_f1))


if __name__ == '__main__':
    print(create_dictionary([['a', 'b'], ['b', 'c']]))
    tune_lstm_hyper(lstm_params={'embedding_dim': 64,
                                 'hidden_dim': 64,
                                 'use_cuda': True,
                                 'device_id': 2})