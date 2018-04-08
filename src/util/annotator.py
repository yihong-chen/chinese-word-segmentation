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


def annotate(input_file, output_file, train=True):
    """Annotate the input file and write it on output_file"""
    f = codecs.open(input_file, 'r', 'utf-8')
    contents = f.read()
    sentences = contents.split(u'\r\n')  # new line for windows
    f.close()

    annotation = []
    for sen in sentences:
        if train:
            # training data can be split by space
            words = sen.split(' ')
        else:
            # test data contains only S
            words = list(sen)
        for word in words:
            if len(word) == 1:
                # single word
                annotation.append(word + cate_of_word(word) + '\tS' + u'\n')
            else:
                # words
                for idx in range(len(word)):
                    tag = word[idx] + cate_of_word(word[idx])
                    if idx == 0:
                        tag += '\tB' + u'\n'
                    elif idx == len(word) - 1:
                        tag += '\tE' + u'\n'
                    else:
                        tag += '\tM' + u'\n'
                    annotation.append(tag)
        annotation.append(u'\n')  # end of a sentence

    f = codecs.open(output_file, 'w', 'utf-8')
    print(len(annotation))
    f.write(''.join(annotation))

    f.close()


if __name__ == '__main__':
    train_file = './data/pku_training.utf8'
    test_file = './data/pku_test.utf8'
    train_dict_file = './data/pku_training_words.utf8'
    annotate(test_file, './test', train=False)

