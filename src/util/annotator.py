import codecs


def cate_of_word(word):
    """return the category of the word"""
    punc = [u'，', u',', u'"', u'“', u'”', u'、', u'：',  u'。', u'！', u'？', u'（', u'）', u'：', u'《', u'》', u'-', u'-', u'%', u'*', u'/', u'.', u'°']
    num = [u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'0',
           u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'○' ,u'零', u'个', u'十', u'百', u'千', u'万', u'亿']
    time = [u'年', u'月', u'日', u'时', u'分', u'秒']

    if word in punc:
        return '\tPUNC'
    elif word in num:
        return '\tNUM'
    elif word in time:
        return '\tTIM'
    else:
        return '\tCN'


def annotate(input_file, output_file=None, train=True, return_sequence=False):
    """Annotate the space-separated input text file i.e. label each word
    write it on output_file or return annotated sentences and labels

    args:
        output_file: where to put the annotated file
        return_sequence: if True return word sequence & label sequence
    """
    f = codecs.open(input_file, 'r', 'utf-8')
    contents = f.read()
    sentences = contents.split(u'\r\n')  # new line for windows
    f.close()
    annotation = []
    if return_sequence:
        word_sequences, label_sequences = [], []
    for sen in sentences:
        if len(sen) == 0:
            continue

        if train:
            # training data can be split by space
            words = sen.split(' ')
        else:
            # test data contains only S
            words = list(sen)

        tag_seq, word_seq = [], []
        for word in words:
            if len(word) == 1:
                # single word
                tag_seq.append('S')
                word_seq.append(word)
                annotation.append(word + cate_of_word(word) + '\tS' + u'\n')
            else:
                # chunk words
                for idx in range(len(word)):
                    word_seq.append(word[idx])
                    tag = word[idx] + cate_of_word(word[idx])
                    if idx == 0:
                        # the first word
                        tag_seq.append('B')
                        tag += '\tB' + u'\n'
                    elif idx == len(word) - 1:
                        # the last word
                        tag_seq.append('E')
                        tag += '\tE' + u'\n'
                    else:
                        # the middle word
                        tag_seq.append('M')
                        tag += '\tM' + u'\n'
                    annotation.append(tag)
        if return_sequence:
            word_sequences.append(word_seq)
            label_sequences.append(tag_seq)
        annotation.append(u'\n')  # end of a sentence

    if output_file is not None:
        f = codecs.open(output_file, 'w', 'utf-8')
        f.write(''.join(annotation))
        f.close()

    if return_sequence:
        print(len(sentences))
        print(len(word_sequences))
        return word_sequences, label_sequences


def annotate_list_of_sents(sentences, labels, output_file=None):
    """

    args:
        output_file: where to put the annotated file
    """
    annotation = []
    for sen, target in zip(sentences, labels):
        for word, tag in zip(sen, target):
            annotation.append(word + cate_of_word(word) + '\tS' + '\t' + tag + u'\r\n')
        annotation.append(u'\r\n')  # end of a sentence

    if output_file is not None:
        f = codecs.open(output_file, 'w', 'utf-8')
        f.write(''.join(annotation))
        f.close()


def segment(input_file, output_file):
    """Convert the annotated text into segmented text
    """
    f = codecs.open(input_file, 'r', 'utf-8')
    contents = f.read()
    lines = contents.split(u'\r\n')
    f.close()

    sentences = []
    sentence = ''
    for line in lines:

        if line == '':
            # end of a sentences
            sentences.append(sentence + u'\n')
            sentence = ''
        else:
            if line == '\n':
                # end of file
                continue
            word, _, _, new_label = line.split()
            if new_label in ['S', 'B']:
                sentence += u' ' + word
            elif new_label in ['M', 'E']:
                sentence += word

    f = codecs.open(output_file, 'w', 'utf-8')
    f.write(''.join(sentences))
    f.close()


if __name__ == '__main__':
    # train_file = './data/pku_training.utf8'
    # test_file = './data/pku_test.utf8'
    # train_dict_file = './data/pku_training_words.utf8'
    # annotate(test_file, './test', train=False)

    crf_test_info = './output_1/crf_test_info.txt'
    segment(crf_test_info, './output_1/crf_test_segmented.txt')
