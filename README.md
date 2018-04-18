# chinese-word-segmentation
Simple chinese word segmentation with experiments on the PKU datatset 
## Methods
- Pattern based word segmentation
- CRF ++ tagging
- LSTM tagging
## Performance
### F1
- Pattern Based Segmentation: 0.87
- CRF++ Tagging: 0.93
- LSTM Tagging: 0.86

It seems that the simple LSTM tagger doesn't perform better than CRF++ or even pattern based segmentation. 

Tips for improve the performance of the LSTM tagger on the segementation task
- Add Tag Inference as stated in the paper [Long Short-Term Memory Neural Networks
for Chinese Word Segmentation
](http://www.aclweb.org/anthology/D15-1141) 
