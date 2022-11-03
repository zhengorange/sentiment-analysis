from transformers import BertModel
import torch.nn as nn


class TextClassification(nn.Module):
    def __init__(self, checkpoint, freeze):
        super(TextClassification, self).__init__()
        self.encoder = BertModel.from_pretrained(checkpoint)
        if freeze == "1":
            self.encoder.embeddings.word_embeddings.requires_grad_(False)
        if freeze == "2":
            self.encoder.embeddings.requires_grad_(False)
        if freeze == "3":
            self.encoder.embeddings.requires_grad_(False)
            self.encoder.encoder.requires_grad_(False)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outs = outs.pooler_output
        logits = self.classifier(outs)
        return logits


if __name__ == '__main__':
    pass
    # mo = TextClassification()
    # print(mo)
