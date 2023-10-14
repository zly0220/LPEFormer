from gensim.models import KeyedVectors,word2vec,Word2Vec

sentences = word2vec.LineSentence('test.txt')
model = Word2Vec(min_count=1,vector_size=128,workers=8)
model.build_vocab(sentences)
model.train(sentences,total_examples=model.corpus_count,epochs=5)
model.save("w2v.model")
# model_load = Word2Vec.load("w2v.model")
# print(model_load.wv['EGDW'].shape)