import os
from util.tuning import report_f_measure, train_dev_split
from util.segment import segment
from util.annotator import annotate


def train(train_file, template_file, crf_param, cache_dir='./output/'):
    """train a crf model uding the template
    args:
        crf_params: dictionary

    return:
        model file dir
    """
    # annotate the input text
    crf_prep_train = os.path.join(cache_dir, 'crf_train_annotated.utf8')
    annotate(train_file, crf_prep_train)
    crf_model = os.path.join(cache_dir, 'crf_model')
    if os.path.exists(crf_model):
        os.remove(crf_model) # clear model cache
    crf_train_info = os.path.join(cache_dir, 'crf_train_info.txt')
    train_cmd = 'crf_learn -f {} -c {} -p {} {} {} {} > {}'.format(crf_param['f'],
                                                                   crf_param['c'],
                                                                   crf_param['p'],
                                                                   template_file,
                                                                   crf_prep_train,  # './output/train.data',#
                                                                   crf_model,
                                                                   crf_train_info)
    os.system(train_cmd)
    return crf_model


def test(crf_model, train_dict_file, test_file, test_file_gold, cache_dir='./output/'):
    crf_prep_test = os.path.join(cache_dir, 'crf_test_annotated.utf8')
    crf_test_info = os.path.join(cache_dir, 'crf_tset_info.txt')
    annotate(input_file=test_file, output_file=crf_prep_test, train=False)
    test_cmd = 'crf_test -m {} {} > {}'.format(crf_model,
                                               crf_prep_test,
                                               crf_test_info)
    os.system(test_cmd)

    # segment the text according to the label
    segmented_test = os.path.join(cache_dir, 'crf_test_segmented.txt')
    segment(crf_test_info, segmented_test)

    # score the result
    score_info = os.path.join(cache_dir, 'crf_score_info.txt')
    score_cmd = 'perl score {} {} {} > {}'.format(train_dict_file,
                                                  test_file_gold,
                                                  segmented_test,
                                                  score_info)
    os.system(score_cmd)

    f_measure = report_f_measure(score_info)
    return f_measure


def tune_crf_hyper(crf_params):

    train_dev_split('./data/pku_training.utf8',
                    './data/pku_tune_training.utf8',
                    './data/pku_tune_training_words.utf8',
                    './data/pku_tune_dev.utf8',
                    './data/pku_tune_dev_gold.utf8')

    results = []
    for c in crf_params['c']:
        for f in crf_params['f']:
            crf_param = {'c': c, 'f': f, 'p': 1}
            crf_model = train(train_file='./data/pku_tune_training.utf8',
                              template_file='./util/crf_template',
                              crf_param=crf_param)
            f_measure = test(crf_model,
                             train_dict_file='./data/pku_tune_training_words.utf8',
                             test_file='./data/pku_tune_dev.utf8',
                             test_file_gold='./data/pku_tune_dev_gold.utf8')
            print((c, f, f_measure))
            results.append((c, f, f_measure))
    return results

if __name__ == '__main__':
    crf_params = {'c': [1, 2, 3, 4, 5],
                  'f': [1, 2, 4, 8, 16, 32, 64]}
    results = tune_crf_hyper(crf_params)
    print(results)