共处理了102980条数据
7种情感数据: [85572, 1022, 353, 174, 12886, 1150, 1823]
other和其他情感: [85572, 17408]
筛选后样本总数:29408
筛选后情感分布: [12000, 1022, 353, 174, 12886, 1150, 1823]
筛选后情感占比: ['40.81%', '3.48%', '1.20%', '0.59%', '43.82%', '3.91%', '6.20%']
训练集大小:23528, 验证集大小:2940, 测试集大小:2940
4) dialogues_emotion.txt: Each line in dialogues_emotion.txt corresponds to the emotion annotations in dialogues_text.txt.
                          The emotion number represents: { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}

在验证集上的NLL损失为: 0.766825, 准确率为: 72.62%
在测试集上的NLL损失为: 0.752931, 准确率为: 72.41%