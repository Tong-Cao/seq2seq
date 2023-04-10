import os
import torch
from d2l import torch as d2l

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    print(os.path.join(data_dir, 'fra.txt'))
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
# print(raw_text[:75])


def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
# print(text[:80])

'''句子分割成一个个单词'''
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]

'''字表 将单词映射到数字'''
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>']) #返回字典
# print('test',src_vocab['hi'])  src_vocab['hi']=2944
len(src_vocab)



"""控制输入文本序列长度一致"""
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

# truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])


#@save
def build_array_nmt(lines, vocab, num_steps):
    """
    将机器翻译的文本序列转换成小批量
    lines:词元化后的文本
    vocab:词表
    num_steps:截断的长度
    """
    lines = [vocab[l] for l in lines] # 将文本转化为词表中的数字
    lines = [l + [vocab['<eos>']] for l in lines] # 每一行加上结束标志符
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len





def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    返回翻译数据集的迭代器和词表
    
    """
    text = preprocess_nmt(read_data_nmt())  # 预处理数据集
    source, target = tokenize_nmt(text, num_examples) # 词元化
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])  # 制作词表
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
    # data_iter(batch_size,num_steps)



# train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
# for X, X_valid_len, Y, Y_valid_len in train_iter:
#     print('X:', X.type(torch.int32))
#     print('X的有效长度:', X_valid_len)
#     print('Y:', Y.type(torch.int32))
#     print('Y的有效长度:', Y_valid_len)
#     break
#输出
# X: tensor([[  6, 143,   4,   3,   1,   1,   1,   1],
#         [ 54,   5,   3,   1,   1,   1,   1,   1]], dtype=torch.int32)
# X的有效长度: tensor([4, 3])
# Y: tensor([[ 6,  0,  4,  3,  1,  1,  1,  1],
#         [93,  5,  3,  1,  1,  1,  1,  1]], dtype=torch.int32)
# Y的有效长度: tensor([4, 3])
