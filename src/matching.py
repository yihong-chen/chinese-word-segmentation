import os
import codecs

from util.pytrie import CharTrie
from util.tuning import report_f_measure


def match(pattern_dict, document, output_document, test_file_gold, cache_dir='./output/'):
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
        idx = 0
        segmented = []
        while idx < len(line):
            matched, _ = pattern_dict_trie.longest_prefix(line[idx:])
            if matched is None:  # if we can not match the character
                matched = line[idx]
            idx += len(matched)
            segmented.append(matched)
        outputs.append(segmented)
    f.close()

    f = codecs.open(output_document, 'w', 'utf-8')
    for line in outputs:
        line = ' '.join(line)
        f.write(line+'\n')
    f.close()

    # score the result
    score_info = os.path.join(cache_dir, 'matching_score_info.txt')
    if os.path.exists(score_info):
        os.remove(score_info) # clear model cache
    score_cmd = 'perl score {} {} {} > {}'.format(train_dict_file,
                                                  test_file_gold,
                                                  output_document,
                                                  score_info)
    os.system(score_cmd)
    f_measure = report_f_measure(score_info)
    return f_measure


if __name__ == '__main__':
    train_file = './data/pku_training.utf8'
    test_file = './data/pku_test.utf8'
    train_dict_file = './data/pku_training_words.utf8'
    test_segmentation_file = './output/matching_test_segmented.txt'
    test_gold_file = './data/pku_test_gold.utf8'
    print('Chinese Word Segmentation by pattern matching ...')
    f1 = match(train_dict_file, test_file, test_segmentation_file, test_gold_file)
    print('F1 measure: {}'.format(f1))