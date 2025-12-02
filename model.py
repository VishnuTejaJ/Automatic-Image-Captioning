import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        # Load a pre-trained ResNet50 model
        # We use ResNet50 because it's powerful but not too heavy.
        self.resnet = models.resnet50(pretrained=True)

        # We want to use the features from the last convolutional layer,
        # not the final classification layer.
        # So we remove the last two layers (avgpool and fc).
        modules = list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # We add a linear layer to convert the ResNet features to our desired embedding size.
        self.embed = nn.Linear(self.resnet[7][-1].conv3.out_channels, embed_size)

    def forward(self, images):
        # Pass images through ResNet
        features = self.resnet(images)  # Shape: (batch_size, 2048, 7, 7)

        # Reshape features to (batch_size, 49, 2048) so we can attend to specific pixels
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(3))

        # Apply the linear layer
        features = self.embed(features)
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # These layers calculate the attention scores
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)

        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)

        # Combine encoder and decoder info to calculate attention
        # We unsqueeze att2 to broadcast it across all pixels
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1)))

        # Calculate alpha (weights) for each pixel
        alpha = self.softmax(att)

        # Apply weights to encoder output to get the context vector
        attention_weighted_encoding = (encoder_out * alpha).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        attention_dim,
        encoder_dim=2048,
        drop_prob=0.3,
    ):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM cell
        # We input: [embedding of current word, context vector from attention]
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)

        # Initial hidden state logic
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        # Final layer to predict the next word
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def init_hidden_state(self, encoder_out):
        # Initialize hidden and cell states using the mean of encoder features
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions):
        # features: (batch_size, num_pixels, encoder_dim)
        # captions: (batch_size, max_length)

        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)

        seq_length = len(captions[0]) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        for s in range(seq_length):
            # Calculate attention context
            context, alpha = self.attention(features, h)

            # Input to LSTM: current word embedding + context
            lstm_input = torch.cat((embeds[:, s], context), dim=1)

            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.dropout(h))

            preds[:, s] = output
            alphas[:, s] = alpha.squeeze(2)

        return preds, alphas


class CNNtoRNN(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, attention_dim, train_CNN=False
    ):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size, train_CNN)
        self.decoderRNN = DecoderRNN(
            embed_size, hidden_size, vocab_size, attention_dim, encoder_dim=embed_size
        )

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs, alphas = self.decoderRNN(features, captions)
        return outputs, alphas

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image.unsqueeze(0)).squeeze(0)
            h, c = self.decoderRNN.init_hidden_state(x.unsqueeze(0))

            # Start with <SOS> token
            input_word = vocabulary.stoi["<SOS>"]

            for _ in range(max_length):
                context, _ = self.decoderRNN.attention(x.unsqueeze(0), h)

                input_embedding = self.decoderRNN.embedding(
                    torch.tensor([input_word]).to(image.device)
                )
                lstm_input = torch.cat((input_embedding, context), dim=1)

                h, c = self.decoderRNN.lstm_cell(lstm_input, (h, c))
                output = self.decoderRNN.fcn(h)

                predicted_word_idx = output.argmax(1).item()
                result_caption.append(predicted_word_idx)

                if vocabulary.itos[predicted_word_idx] == "<EOS>":
                    break

                input_word = predicted_word_idx

        return [vocabulary.itos[idx] for idx in result_caption]
