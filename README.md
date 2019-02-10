# தமிழ் மொழி ஒப்பு - tamil-lm
மொழியொப்பேற்றம், வெறும் செய்திதாள் கட்டுரைகளிலிருந்து, கணினி எப்படி சொற்களுக்கு இருக்கும் தொடர்பை கண்டுபிடிக்கிறது.
Language modelling based on skip-grams over tamil news dataset

## [காட்சிப்பொருள் - Demo](http://w2v.kaatchi.cheyyarivu.org/)
இந்த தளத்தில் சென்று பார்க்க, தொடர்புடைய சொற்கள் எவையவை, தொடர்பில்லாத சொற்கள் எவையவை என்று கணியொப்பே கண்டறிந்து காட்டுகிறது புரியும்.
Please go to the [link](http://w2v.kaatchi.cheyyarivu.org/) for the demo.

## தரவுக்கணம் - Dataset
The data is scraped from tamil news websites. The dataset will be made available soon. 

## Models
Though there are three models available in the repo, skipgram works well.

## Model weights.
### Plain text vocabulary and embedding vectors
The embedding vectors and corresponding tokens can be downloaded from [vaaku2vec.zip](https://drive.google.com/open?id=1G3FM2paj9JaX-zsg0yDWxAHGrlxnjROy)
You can upload the vocab.vectors.tsv and vocab.tokens.tsv in [TensorFlow Projector](projector.tensorflow.org) to visualize them. Use TSNE projection and let it run for more than 400 iterations. You can see the cone-ice shape come to life. It is really fun. 

## Training
    $ python main.py train
  
