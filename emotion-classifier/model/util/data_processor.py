from data_iterator import DataIterator
import random


class DataProcessor(object):
    r""" 实现数据的预处理 """
    def __init__(self, data, batch_size, sp, shuffle=True):
        self.sp = sp
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_batch_data(self):
        r""" 输出一个batch预处理的样本 """
        if self.shuffle:
            random.shuffle(self.data)
        it = DataIterator(self.data, self.batch_size)

        for batch_data in it.get_batch_data():
            texts, emotions = [], []
            for item in batch_data:
                texts.append(item['text'].strip().split())
                emotions.append(item['emotion'])

            id_texts, len_texts = [], []
            for text in texts:
                id_text, len_text = self.sp.word2index(text)
                id_texts.append(id_text)
                len_texts.append(len_text)

            len_texts = [l+2 for l in len_texts]  # 加上start和end后的长度
            maxlen_text = max(len_texts)

            pad_id_texts = [self.sp.pad_sentence(t, maxlen_text) for t in id_texts]  # 补齐长度

            new_batch_data = {'str_texts': texts,  # [batch, len]
                              'texts': pad_id_texts,  # [batch, len]
                              'len_texts': len_texts,  # [batch]
                              'emotions': emotions}  # [batch]

            yield new_batch_data
