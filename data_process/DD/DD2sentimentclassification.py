import argparse
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--text_path', dest='text_path', default='../../ijcnlp_dailydialog/dialogues_text.txt', type=str, help='文本路径')
parser.add_argument('--emo_path', dest='emo_path', default='../../ijcnlp_dailydialog/dialogues_emotion.txt', type=str, help='情感路径')
parser.add_argument('--save_path', dest='save_path', default='./result', type=str, help='保存路径')
args = parser.parse_args()


def dd2_sentiment_classification(text_path, emo_path, save_path):
    if not os.path.isfile(text_path) or not os.path.isfile(emo_path):
        print('请输入正确的数据路径！')
        return
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_name = 'SentimentClassification.txt'
    save_path = os.path.join(save_path, save_name)
    num = 0

    ft = open(text_path, 'r', encoding='utf8')
    fe = open(emo_path, 'r', encoding='utf8')
    fs = open(save_path, 'w', encoding='utf8')

    for ss, texts in enumerate(ft):  # 原数据集第673句买自行车的少一个标签
        emotions = fe.readline().strip().split()
        texts = texts.strip().split('__eou__')[:-1]
        assert len(texts) == len(emotions)
        for idx, text in enumerate(texts):
            fs.write(json.dumps({'text': text, 'emotion': int(emotions[idx])}, ensure_ascii=False)+'\n')
            num += 1
    print(f'共处理了{num}条数据')
    ft.close()
    fe.close()
    fs.close()

    emo_num = [0] * 7
    with open(save_path, 'r', encoding='utf8') as fr:
        for line in fr:
            data = json.loads(line.strip())
            assert 0 <= data['emotion'] <= 6
            emo_num[data['emotion']] += 1
    print('7种情感数据:', emo_num)
    print('other和其他情感:', [emo_num[0], sum(emo_num[1:])])


if __name__ == '__main__':
    dd2_sentiment_classification(args.text_path, args.emo_path, args.save_path)
