import os
import shutil
from util.tuning import report_f_measure, train_dev_split
from util.annotator import annotate, segment


def train(train_file, template_file, crf_param, cache_dir='./output/'):
    """train a crf model uding the template
    args:
        crf_params: dictionary

    return:
        model file dir
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # annotate the input text
    crf_prep_train = os.path.join(cache_dir, 'crf_train_annotated.utf8')
    annotate(train_file, crf_prep_train)
    crf_model = os.path.join(cache_dir, 'crf_model')
    if os.path.exists(crf_model):
        os.remove(crf_model) # clear model cache
    crf_train_info = os.path.join(cache_dir, 'crf_train_info.txt')
    if os.path.exists(crf_train_info):
        os.remove(crf_train_info) # clear model cache
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
    crf_test_info = os.path.join(cache_dir, 'crf_test_info.txt')
    if os.path.exists(crf_test_info):
        os.remove(crf_test_info) # clear model cache
    annotate(input_file=test_file, output_file=crf_prep_test, train=False)
    test_cmd = 'crf_test -m {} {} > {}'.format(crf_model,
                                               crf_prep_test,
                                               crf_test_info)
    os.system(test_cmd)

    # segment the text according to the label
    segmented_test = os.path.join(cache_dir, 'crf_test_segmented.txt')
    if os.path.exists(segmented_test):
        os.remove(segmented_test) # clear model cache
    segment(crf_test_info, segmented_test)

    # score the result
    score_info = os.path.join(cache_dir, 'crf_score_info.txt')
    if os.path.exists(score_info):
        os.remove(score_info) # clear model cache
    score_cmd = 'perl score {} {} {} > {}'.format(train_dict_file,
                                                  test_file_gold,
                                                  segmented_test,
                                                  score_info)
    os.system(score_cmd)
    f_measure = report_f_measure(score_info)
    return f_measure


def tune_crf_hyper(crf_params, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # train_dev_split('./data/pku_training.utf8',
    #                 './data/pku_tune_training.utf8',
    #                 './data/pku_tune_training_words.utf8',
    #                 './data/pku_tune_dev.utf8',
    #                 './data/pku_tune_dev_gold.utf8')

    results = []
    for c in crf_params['c']:
        for f in crf_params['f']:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)

            crf_param = {'c': c, 'f': f, 'p': 50}
            crf_model = train(train_file='./data/pku_tune_training.utf8',
                              template_file='./util/crf_template',
                              crf_param=crf_param,
                              cache_dir=cache_dir)
            f_measure = test(crf_model,
                             train_dict_file='./data/pku_tune_training_words.utf8',
                             test_file='./data/pku_tune_dev.utf8',
                             test_file_gold='./data/pku_tune_dev_gold.utf8',
                             cache_dir=cache_dir)
            print((c, f, f_measure))
            results.append((c, f, f_measure))
    return results


if __name__ == '__main__':
    # crf_params = {'c': [1, 2, 3, 4, 5],
    #               'f': [20, 15, 10, 5]}
    # results = tune_crf_hyper(crf_params, cache_dir='./output_2')
    # print(results)

    crf_param = {'c': 1, 'f': 5, 'p': 50}
    crf_model = train(train_file='./data/pku_training.utf8',
                      template_file='./util/crf_template',
                      crf_param=crf_param,
                      cache_dir='./output_1/')
    f_measure = test(crf_model,
                     train_dict_file='./data/pku_training_words.utf8',
                     test_file='./data/pku_test.utf8',
                     test_file_gold='./data/pku_test_gold.utf8',
                     cache_dir='./output_1/')
    print(f_measure)