import codecs

from utils.pytrie import CharTrie


def match(pattern_dict, document, output_document):
    """Segment document based on the pattern_dict, write the segmented file in output_document"""
    f = codecs.open(pattern_dict, 'r', 'utf-8')
    pattern_dict = [line.strip() for line in f.readlines()]
    pattern_dict.sort(key=len, reverse=True)
    max_len = len(pattern_dict[0])
    print('Length of the dictionary: {}; Max length of words: {}'.format(len(pattern_dict), max_len))
    f.close()

    pattern_dict_trie = CharTrie()
    for pattern in pattern_dict:
        pattern_dict_trie[pattern] = True

    f = codecs.open(document, 'r', 'utf-8')
    outputs = []
    for line in f.readlines():
        line = line.strip()
        print(line)
        idx = 0
        segmented = []
        while idx < len(line):
            matched, _ = pattern_dict_trie.longest_prefix(line[idx:])
            if matched is None:  # if we can not match the character
                matched = line[idx]
            idx += len(matched)
            print(matched)
            segmented.append(matched)
        outputs.append(segmented)
    f.close()

    f = codecs.open(output_document, 'w', 'utf-8')
    for line in outputs:
        line = ' '.join(line)
        f.write(line+'\n')
    f.close()


if __name__ == '__main__':
    train_file = './data/pku_training.utf8'
    test_file = './data/pku_test.utf8'
    train_dict_file = './data/pku_training_words.utf8'
    test_segmentation_file = './output/matching_pku_test_seg.utf8'
    print('Chinese Word Segmentation by pattern matching ...')
    match(train_dict_file, test_file, test_segmentation_file)