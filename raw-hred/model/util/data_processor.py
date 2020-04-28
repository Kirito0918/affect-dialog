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
            str_posts, str_responses = [], []  # post和response的str表示

            for item in batch_data:
                str_posts.append(item['posts'])  # [batch, turn, len]
                str_responses.append(item['response'])  # [batch, len]

            id_posts, id_responses, len_posts, len_responses, turn_posts = [], [], [], [], []

            for posts in str_posts:
                turn_posts.append(len(posts))  # [batch] 记录每个样本有多少个post
                id_post = []  # [turn, len] 记录一个样本每个post的id表示
                len_post = []  # [turn] 记录一个样本每个post的长度
                for post in posts:
                    idx, length = self.sp.word2index(post)
                    id_post.append(idx)
                    len_post.append(length)
                id_posts.append(id_post)  # [batch, turn, len]
                len_posts.append(len_post)  # [batch, turn]

            for response in str_responses:  # response从str2index并统计长度
                id_response, len_response = self.sp.word2index(response)
                id_responses.append(id_response)  # [batch, len]
                len_responses.append(len_response)  # [batch]

            len_posts = [[u+2 for u in b]for b in len_posts]  # 加上start和end后的长度
            len_responses = [l+2 for l in len_responses]
            maxlen_turn = max(turn_posts)
            maxlen_post = max([max(_) for _ in len_posts])
            maxlen_response = max(len_responses)

            pad_len_posts = []
            for len_post in len_posts:
                if len(len_post) < maxlen_turn:
                    for _ in range(maxlen_turn - len(len_post)):
                        len_post.append(2)
                pad_len_posts.append(len_post)
            pad_id_posts = []
            temp = [[self.sp.pad_sentence(p, maxlen_post) for p in b] for b in id_posts]  # 补齐长度 [batch, turn, maxlen]
            for posts in temp:  # [turn, len]
                if len(posts) < maxlen_turn:
                    for _ in range(maxlen_turn-len(posts)):
                        posts.append(self.sp.get_pad_sentence(maxlen_post))
                pad_id_posts.append(posts)  # [batch, turn, maxlen]
            pad_id_responses = [self.sp.pad_sentence(r, maxlen_response) for r in id_responses]

            new_batch_data = {'str_posts': str_posts,
                              'str_responses': str_responses,
                              'posts': pad_id_posts,  # [batch, maxturn, maxlen]
                              'responses': pad_id_responses,  # [batch, maxlen]
                              'len_posts': pad_len_posts,  # [batch, maxturn]
                              'len_responses': len_responses,
                              'turn_posts': turn_posts}  # [batch]

            yield new_batch_data
