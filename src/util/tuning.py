import torch
import codecs
from sklearn.cross_validation import train_test_split


def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def write_sentences(file, sentences):
    """Write a list of sentences into file"""
    f = codecs.open(file, 'w', 'utf-8')
    f.write('\r\n'.join(sentences))
    f.close()


def train_dev_split(input_file, train_file, train_dict_file, dev_file, dev_gold_file):
    f = codecs.open(input_file, 'r', 'utf-8')
    contents = f.read()
    f.close()
    sentences = contents.split('\r\n')
    train_sentences, dev_sentences = train_test_split(sentences, test_size=0.1)
    print('Num of training sentences: {}\nNum of developing sentences: {}'.format(len(train_sentences), len(dev_sentences)))

    write_sentences(train_file, train_sentences)
    write_sentences(dev_gold_file, dev_sentences)

    # generate dictionary for train data
    train_dict = []
    for sentence in train_sentences:
        words = sentence.split(' ')
        for word in words:
            if len(word) > 0:
                train_dict.append(word)
    write_sentences(train_dict_file, train_dict)

    # rejoin the dev gold file as a dev set
    dev_sentences = [sentence.replace(' ', '') for sentence in dev_sentences]
    write_sentences(dev_file, dev_sentences)


def report_f_measure(score_info_file):
    f_measure = None
    for codec in ('utf-8', 'gb2312'):
        try:
            f = codecs.open(score_info_file, 'r', codec)
            contents = f.read()
            sentences = contents.split('\n')
            for sentence in sentences:
                if sentence.startswith('=== F MEASURE:'):
                    f_measure = float(sentence.split(':')[-1])
                    break
        except UnicodeDecodeError:
            pass
    if f_measure is None:
        raise UnicodeDecodeError("{}".format(score_info_file), )
    return f_measure


if __name__ == '__main__':
    train_dev_split('./data/pku_training.utf8',
                    './data/pku_tune_training.utf8',
                    './data/pku_tune_training_words.utf8',
                    './data/pku_tune_dev.utf8',
                    './data/pku_tune_dev_gold.utf8')
    # print(report_f_measure('./output/crf_score_info.txt'))