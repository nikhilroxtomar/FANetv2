import numpy as np
from bpemb import BPEmb

""" Subword Embeddings: https://nlp.h-its.org/bpemb """

class Text2Embed:
    def __init__(self, max_length=16):
        self.max_length = max_length
        self.bpemb_en = BPEmb(lang="en", vs=100000, dim=300)

    def to_tokens(self, word):
        tokens = self.bpemb_en.encode(word)
        return tokens

    def to_embed(self, sentence):
        embed = self.bpemb_en.embed(sentence)
        if embed.shape[0] < self.max_length:
            zero_embed = np.zeros((self.max_length - embed.shape[0], 300))
            embed = np.concatenate([embed, zero_embed], axis=0)

        return embed

if __name__ == "__main__":
    sentence = " colorectal image with many small medium large sized polyps"
    embed = Text2Embed()

    tokens = embed.to_tokens(sentence)
    vec = embed.to_embed(sentence)
    print(vec.shape)
