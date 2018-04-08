import codecs


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
            word, _, _, new_label = line.split()
            if new_label in ['S', 'B']:
                sentence += u' ' + word
            elif new_label in ['M', 'E']:
                sentence += word

    f = codecs.open(output_file, 'w', 'utf-8')
    f.write(''.join(sentences))
    f.close()


if __name__ == '__main__':
    crf_test_info = './output/crf_test_info.txt'
    segment(crf_test_info, './output/crf_test_segmented.txt')