import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


paragraph = """Winter is coming, and the great continent of Westeros stands on 
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

sentence = nltk.sent_tokenize(paragraph)
words = nltk.word_tokenize(paragraph)

print("\n STEMMING: stemming is a process of cutting down words into by removing the suffix. \n")
print("\n Words before stemming: \n")
print(words)


stemmer = PorterStemmer()
print("\n Words after stemming: \n")
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    words = [stemmer.stem(word) for word in words if word.lower() not in stopwords.words('english')]
    sentence[i] = " ".join(words)
    print(sentence[i])
