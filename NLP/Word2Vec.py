import re
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.corpus import stopwords
from gensim.models import Word2Vec

pararaph = """
Winter is coming, and the great continent of Westeros stands on 
the edge of chaos. In the North, House Stark rules from Winterfell with loyalty 
and honor, but their peaceful life begins to break when they are pulled into 
the dangerous politics of the capital. Far beyond the Wall, strange and ancient 
forces awaken in the frozen wilderness, threatening the realm with an enemy 
that cannot be defeated by swords alone. In King’s Landing, House Lannister 
holds power through wealth and manipulation, while whispers of betrayal move 
through the royal court like poison. Across the Narrow Sea, the last surviving 
Targaryen heirs dream of reclaiming the Iron Throne, gathering strength 
and allies as dragons rise once again. As rival houses form alliances
 and break them just as quickly, every decision becomes a matter of life 
 and death, and every secret has the power to destroy kingdoms. In this world, 
 even the strongest warriors fall, the wisest leaders are deceived, 
 and the battle for power leaves no one untouched."""

text = re.sub(r'\[[0-9]"\]', " ", pararaph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()

sentences = nltk.sent_tokenize(pararaph)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

print(sentences)

#training model for Word2Vec

model = Word2Vec(sentences, min_count=1)
print(model)

words = model.build_vocab_from_freq

#finding word vectors
vector = model.wv['dragons']
print(vector)

# most similar words
similar = model.wv.most_similar('dragons')
print(similar)
