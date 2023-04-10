import math
import pandas as pd
import torch
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from d2l import torch as d2l


class PositionalEncoding(nn.Module):
    """位置编码

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
                0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 创建transformer模型


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_hiddens,  num_heads,
                 num_layers, dropout, **kwargs):
        """
        初始化模型参数
        src_vocab_size: encode输入词典大小
        tgt_vocab_size: decode输入词典大小
        embed_size: 词嵌入大小
        num_hiddens: feedforward隐藏层大小  默认值为2048
        num_heads: 多头注意力中的头数
        num_layers: 编码器和解码器的层数
        dropout: 丢弃率
        """
        super(TransformerModel, self).__init__(**kwargs)

        self.embed_size = embed_size
        # encode数据词嵌入层 (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        # decode数据词嵌入层 (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

        # 位置编码层
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        # transformer 输入输出维度都是embed_size
        self.transformer = nn.Transformer(d_model=embed_size, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=num_hiddens,
                                          dropout=dropout, nhead=num_heads,
                                          batch_first=True)
        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(embed_size, tgt_vocab_size)

    def forward(self, src, tgt):
        """
        前向传播
        src: 喂给encoder的 sequence 形状(batch_size, seq_len)
        tgt: 喂给decoder的 sequence 形状(batch_size, seq_len)
        """
        # tgt_mask:decode的输入词源使其在训练时看不到后面的信息
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        # src_key_padding_mask:encode的输入词源使其在训练时看不到pad的信息
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src).to(device)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt).to(device)

        # 对词源进行嵌入和位置编码 embedding后的数值较小将其乘上embed_size方根适当放大
        src = self.pos_encoding(self.src_embedding(
            src) * math.sqrt(self.embed_size))
        # src: (batch_size, seq_len, embed_size)
        tgt = self.pos_encoding(self.tgt_embedding(
            tgt) * math.sqrt(self.embed_size))
        # tgt: (batch_size, seq_len, embed_size)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        # out: (batch_size, seq_len, embed_size)
        out = self.predictor(out)  # out: (batch_size, seq_len, tgt_vocab_size)

        return out  # (batch_size, seq_len, vocab_size)

    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask 将所有
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == src_vocab['<pad>']] = -torch.inf 
        #根据词典src_vocab选择将pad的位置设为-inf
        return key_padding_mask


'''模型测试'''
# src = torch.LongTensor([[0, 3, 4, 5, 6, 1, 2, 2]])
# tgt = torch.LongTensor([[3, 4, 5, 6, 1, 2, 2, 1]])

# model = TransformerModel(vocab_size=10, embed_size=4, num_hiddens=128, num_heads=2, num_layers=1, dropout=0.0)
# out = model(src, tgt)
# print(out.size()) # torch.Size([1, 8, 10])
# print(out)

def train(net, train_iter, lr, num_epochs, tgt_vocab, device):

    total_loss = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用带掩码的交叉熵损失函数
    loss = d2l.MaskedSoftmaxCELoss()
    net.train()
    net.to(device)

    for step in range(num_epochs):
        for batch in train_iter:
            # 清空梯度
            optimizer.zero_grad()

            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 强制教学 将decoder的输入加上bos
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)

            out = net(X, dec_input)
            # 掩码的交叉熵损失函数 Y_valid_len存放的是每个句子的有效长度
            l = loss(out, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)  # 梯度剪裁防止梯度爆炸

            optimizer.step()

            total_loss += l.sum().item()
            # 每40次打印一下loss
        if step != 0 and step % 40 == 0:
            print("Step {}, total_loss: {}".format(step, total_loss))
            total_loss = 0

#预测

def predict(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    net.to(device)
    # 将句子分割成词
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    
    # 将句子补全到num_steps
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
     
    # 将索引转换成tensro 并添加维度变成二维 
    src_seq = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
    
    # 生成bos
    bos = torch.tensor([tgt_vocab['<bos>']], device=device).view(1, 1)
    # 将bos加入到decoder的输入中
    dec_seq = bos

    for _ in range(num_steps):
        # 进行预测
        out = net(src_seq, dec_seq)
        # 选择最后一个词
        out = out[:, -1, :]
        # 选择概率最大的词
        pred = out.argmax(dim=1).reshape(1, 1)
        # 将预测的词加入到decoder的输入中
        dec_seq = torch.cat([dec_seq, pred], dim=1)
        # 更新decoder的mask
        dec_valid_len = torch.tensor([dec_seq.shape[1]], device=device)
        # 如果预测的词是eos则结束预测
        if pred.item() == tgt_vocab['<eos>']:
            break

    # 将预测的索引转换成词
    tokens = [tgt_vocab.idx_to_token[i] for i in dec_seq[0].cpu().numpy()]
    # 去掉bos和eos
    return tokens[1:-1]


if __name__ == '__main__':
    # num_steps为输入文本的截断长度，batch_size为每次迭代的批量大小
    batch_size, num_steps = 64, 10
    embed_size, num_hiddens, num_heads, dropout,  = 32, 128, 4, 0.1
    lr, num_epochs, device = 0.005, 500, d2l.try_gpu()
    # 加载数据集
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    # 定义模型
    net = TransformerModel(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), embed_size=embed_size,
                           num_hiddens=num_hiddens, num_heads=num_heads, num_layers=2, dropout=dropout)
    
    train(net, train_iter, lr, num_epochs, tgt_vocab, device)

    src_sentence ='go .'
    predict_result = predict(net, src_sentence, src_vocab, tgt_vocab, num_steps, device)
    print('预测结果',predict_result)



   
