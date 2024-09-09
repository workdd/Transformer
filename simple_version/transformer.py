import torch
import torch.nn as nn
import torch.nn.functional as F

import math

max_seq_length = 100
total_word_num = 100


class Transformer(nn.Module):
    def __init__(self, encoder_num=6, decoder_num=6, hidden_dim=512, max_encoder_seq_length=100, max_decoder_seq_length=100):
        super().__init__()

        self.encoder_num = encoder_num
        self.decoder_num = decoder_num

        self.hidden_dim = hidden_dim
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length

        self.input_data_embed = nn.Embedding(total_word_num, self.hidden_dim)
        self.Encoders = [Encoder(dim_num=hidden_dim) for _ in range(encoder_num)]

        self.output_data_embed = nn.Embedding(total_word_num, self.hidden_dim)
        self.Decoders = [Decoder(dim_num=hidden_dim) for _ in range(decoder_num)]

        self.last_linear_layer = nn.Linear(self.hidden_dim, max_seq_length)

    def positional_encoding(self, position_max_length=100):
        position = torch.arange(0, position_max_length, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(position_max_length, self.hidden_dim)
        div_term = torch.pow(
            torch.ones(self.hidden_dim // 2).fill_(10000), torch.arange(0, self.hidden_dim, 2) / torch.tensor(self.hidden_dim, dtype=torch.float32)
        )

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)

        # 모델 학습 시 positional encoding은 학습되지 않도록 설정
        self.register_buffer("pe", pe)

        return pe

    def forward(self, input, output, mask):
        input_embed = self.input_data_embed(input)
        input_embed += self.positional_encoding(self.max_encoder_seq_length)
        query, key, value = input_embed, input_embed, input_embed

        for encoder in self.Encoders:
            encoder_output = encoder(query, key, value)
            query = encoder_output
            key = encoder_output
            value = encoder_output

        output_embed = self.output_data_embed(output)
        output += self.positional_encoding(self.max_decoder_seq_length)
        masked_unsqueezed = mask.unsqueeze(-1)
        output_embed = output_embed.masked_fill(masked_unsqueezed, 0)
        d_query, d_key, d_value = output_embed, output_embed, output_embed

        decoder_output = None
        for decoder in self.Decoders:
            decoder_output = decoder(d_query, d_key, d_value)
            d_query = decoder_output
            d_key = decoder_output
            d_value = decoder_output

        output = F.softmax(self.last_linear_layer(decoder_output), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_num=512, head_num=8) -> None:
        super().__init__()
        self.head = head_num
        self.dim_num = dim_num

        self.query_embed = nn.Linear(dim_num, dim_num)
        self.key_embed = nn.Linear(dim_num, dim_num)
        self.value_embed = nn.Linear(dim_num, dim_num)
        self.output_embed = nn.Linear(dim_num, dim_num)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = key.size()[-1]
        key_transpose = torch.transpose(key, 3, 2)  ## (3,2) ??

        output = torch.amtmul(query, key_transpose)
        output /= math.sqrt(d_k)

        if mask:
            mask_unsqueezed = mask.unsqueeze(1).unsqueeze(-1, 0)
            output = output.masked_fill(mask_unsqueezed)

        output = F.softmax(output, -1)
        output = torch.matmul(output, value)

        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size()[0]

        query = self.query_embed(query).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)
        key = self.key_embed(key).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)
        value = self.value_embed(value).view(batch_size, -1, self.head_num, self.dim_num // self.head_num).transpose(1, 2)

        output = self.scaled_dot_product_attention(query, key, value, mask)
        batch_num, head_num, seq_num, hidden_num = output.size()
        output = torch.transpose(output, 1, 2).contiguous().view((batch_size, -1, hidden_num * self.head_num))

        return output


class AddLayerNormalization(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def layer_normalization(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.std(input, dim=-1, keepdim=True)
        output = (input - mean) / std

        return output

    def forward(self, input, residual):
        return residual + self.layer_normalization(input)


class FeedForward(nn.Module):
    def __init__(self, dim_num=512) -> None:
        super().__init__()
        self.layer1 = nn.Linear(dim_num, dim_num * 4)
        self.layer2 = nn.Linear(dim_num * 4, dim_num)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(F.relu(output))

        return output


class Encoder(nn.Module):
    def __init__(self, dim_num=512) -> None:
        super().__init__()

        self.multihead = MultiHeadAttention(dim_num=dim_num)
        self.residual_layer1 = AddLayerNormalization()
        self.feedforward = FeedForward(dim_num=dim_num)
        self.residual_layer2 = AddLayerNormalization()

    def forward(self, query, key, value):
        multihead_output = self.multihead(query, key, value)
        residual1_output = self.residual_layer1(multihead_output, query)
        feedforward_output = self.feedforward(residual1_output)
        output = self.residual_layer2(feedforward_output, residual1_output)

        return output


class Decoder(nn.Module):
    def __init__(self, dim_num=512) -> None:
        super().__init__()

        self.masked_multihead = MultiHeadAttention(dim_num=dim_num)
        self.residual_layer1 = AddLayerNormalization()
        self.multihead = MultiHeadAttention(dim_num=dim_num)
        self.residual_layer2 = AddLayerNormalization()
        self.feed_forward = FeedForward(dim_num=dim_num)
        self.residual_layer3 = AddLayerNormalization()

    def forward(self, o_query, o_key, o_value, encoder_output, mask):
        masked_multihead_output = self.masked_multihead(o_query, o_key, o_value, mask)
        residual1_output = self.residual_layer1(masked_multihead_output, o_query)
        multihead_output = self.multihead(encoder_output, encoder_output, residual1_output, mask)
        residual2_output = self.residual_layer2(multihead_output, residual1_output)
        feedforward_output = self.feed_forward(residual2_output)
        output = self.residual_layer3(feedforward_output, residual2_output)


if __name__ == "__main__":
    model = Transformer()

    input = torch.randint(low=0, high=max_seq_length, size=(64, max_seq_length), dtype=torch.long)
    output = torch.randint(low=0, high=max_seq_length, size=(64, max_seq_length), dtype=torch.long)
    mask = torch.zeros(64, max_seq_length)
    mask[:, :30] = 1

    output = model(input, output, mask)
    _, pred = torch.max(output, dim=-1)
    print(pred.shape)
