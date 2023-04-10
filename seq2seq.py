import collections
import math
import torch
from torch import nn
import numpy as np
from d2l import torch as d2l


"""用于序列到序列学习的循环神经网络编码器"""


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 使用embadding将字表的表示压缩 vocab_size:字典的大小 embed_size：编码后的长度
        # imput=（batch_size,num_steps） output=(batch_size,num_steps,embed_size)

        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)  # GRU作为编码器  num_layers=2为设置进过两层GRU再输出
        # imput=(batch_size,num_steps,embed_size ) output=(batch_size,num_steps,num_hiddens )
        # embed_size输入的size   num_hiddens为输出的size

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


"""用于序列到序列学习的循环神经网络解码器"""


class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]  # 取encode的最后一个隐藏状态作为decode的初始状态

    def forward(self, X, state):
        # embedding输出'X'的形状：(batch_size,num_steps,embed_size)
        # x：(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        # state:(num_laylar,batch_size,num_hiddens) state[-1]:(batch_size,num_hiddens)最后一层GRU的最后隐藏状态
        # .repeat(X.shape[0], 1, 1)    X.shape[0]=num_steps  将state[-1]在通道上复制num_steps个
        # context：(num_steps,batch_size,num_hiddens)
        # X_and_context：(num_steps,batch_size,num_hiddens+embed_size)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    #  x=[[1,2,3],[4,5,6]] (2,3)
    #  valid_len=[1,2] 分别对应每一排序列需要保留的长度
    maxlen = X.size(1)  # x的第二维度大小X.size(1)=3  通常为 max_sequence_length
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # torch.arange((maxlen=3), dtype=torch.float32,device=X.device): 一维序列 [None, :]变为二维[[0,1,2]]
    # valid_len=[1,2]一维度   变为二维valid_len[:, None]=[[1],[2]]
    #  < 逐元素比较 [[0,1,2]]和[[1],[2]]用到广播机制   [[0<1,1<1,2<1], [0<2 ,1<2,2<2]] 最后mask为
    #                                               [[true,false,false], [true,true,false]]
    X[~mask] = value
    # 先将mask取反[[false,true,true], [false ,true,true]]
    # 最后与x比较 [[1,2,3],[4,5,6]]
    # 将取反后的mask中为ture的值 相同位置的x中的值变为value=0
    # X = [[1,0,0],[4,5,0]]
    return X

# @save


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred 模型预测值的形状：(batch_size,num_steps,vocab_size)
    # label 标签的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)  valid_len=[2,3,4]分别为每个语句的有效长度

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)  # 生成和label相同的张量并将值都设置为1
        # 根据valid_len使用sequence_mask进行将weights无效部分设置为0
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)  # 使用父类nn.CrossEntropyLoss正常计算损失值
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        # 使用weights来对loss加权（将无效的部分乘以0）并沿着维度 1 取平均值，得到最终的加权损失值。
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    def xavier_init_weights(m):
        '''初始化模型参数'''
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器
    loss = MaskedSoftmaxCELoss()  # 带遮蔽的softmax交叉熵损失函数
    net.train()  # 开启反向传播
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])  # 在动画中绘制数据
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)  # 给decode的输入加入开始标志
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 在Y前加上开始标志 并且所有序列往后移一位 Y[:, :-1]相当于去掉最后一列
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)  # 梯度剪裁防止梯度爆炸
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


if __name__ == '__main__':
  embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
  batch_size, num_steps = 64, 10
  lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

  train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
  encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
  decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
  net = d2l.EncoderDecoder(encoder, decoder)
  train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

  src_sentence ='go .'
  x,y = predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False)
  print(x)


