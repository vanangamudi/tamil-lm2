# tamil-news-classification
Classification of tamil news headlines - experimental

## Data source
The data is scraped from puthiyathalaimurai.com. The model accuracy is little over 60 percent. Though, we use only the headlines of the news. Using the article content or part of it might improve the accuracy. 

## Training
    $ python main.py train
  
## Testing
### Prediction
    (torch) ~/projects/text_classification/main$ head ../dataset/text.subword_nmt.txt  | python main.py predict 
    Namespace(hpconfig='hpconfig.py', log_filter=None, save_plot=False, show_plot=False, task='predict')
    INFO    :anikattu.utilz.initialize_tasks>> loading hyperparameters from hpconfig.py
    ====================================
    99f4a4
    ====================================
    INFO    :__main__.<module>s>> flushing...
    INFO    :utilz   .load_datas>> processing file: ('../dataset/text.subword_nmt.txt', '../dataset/label.txt')
    processing ('../dataset/text.subword_nmt.txt', '../dataset/label.txt'): 10200it [00:00, 258355.73it/s]
    skipped 0 samples
    INFO    :utilz   .load_datas>> building input_vocabulary...
    INFO    :anikattu.vocab.__init__s>> Constructiong vocabuluary object...
    INFO    :anikattu.vocab.__init__s>> number of word in index2word and word2index: 667 and 667
    INFO    :anikattu.vocab.__init__s>> Constructiong vocabuluary object...
    INFO    :anikattu.vocab.__init__s>> number of word in index2word and word2index: 6 and 6
    INFO    :anikattu.dataset.__init__s>> building dataset: ('../dataset/text.subword_nmt.txt', '../dataset/label.txt')
    INFO    :anikattu.dataset.__init__s>> build dataset: ('../dataset/text.subword_nmt.txt', '../dataset/label.txt')
    INFO    :anikattu.dataset.__init__s>>  trainset size: 8194
    INFO    :anikattu.dataset.__init__s>>  testset size: 911
    INFO    :anikattu.dataset.__init__s>>  input_vocab size: 667
    INFO    :anikattu.dataset.__init__s>>  output_vocab size: 6
    INFO    :__main__.<module>s>> dataset size: 8194
    INFO    :__main__.<module>s>> vocab: Counter({'tamilnadu': 3115,
             'india': 2263,
             'cinema': 1256,
             'sports': 1057,
             'world': 712,
             'politics': 702})
    INFO    :__main__.<module>s>> loaded the old image for the model from :99f4a4/weights/main.pth
    **** the model Model(
      (embed): Embedding(667, 300)
      (encode): LSTM(300, 300, bidirectional=True)
      (classify): Linear(in_features=600, out_features=6, bias=True)
    )
    =========== PREDICTION ==============
    ?“நேர்மையான கிரிக்கெட்டை விளையாட தென் இந்தியா என்னை தயார்ப்படுத்தியது” - தோனி == sports
    ?மேகதாது விவகாரம்: தமிழக, கர்நாடகா முதலமைச்சர்களுக்கு நிதின் கட்கரி கடிதம் == india
    ?உண்மை நிலை தெரியாமல் பதிலளிக்க முடியாது - நடிகர் ரஜினிகாந்த் == cinema
    ?“தமிழகத்தின் அனுமதி இல்லாமல் மேகதாது அணை கட்ட முடியாது”- நிதின் கட்கரி..! == india
    ?“பந்துவீச்சாளர்கள் ஐபிஎல் விளையாடலாமா?” - எதிரெதிர் கருத்தில் தோனி, கும்பளே  == sports
    ?ஜான்சன் அன்ட் ஜான்சன் பவுடரை ஆய்வு செய்ய மத்திய அரசு அறிவுறுத்தல் == india
    ?தமிழகத்தில் 2 தினங்களுக்கு மழைக்கு வாய்ப்பு : வானிலை மையம் தகவல் == tamilnadu
    ?சிறுத்தையை கூண்டு வைத்து பிடித்தாலும் பிரச்னை முடியாது ! == tamilnadu
    ?'நானும்தான் ஆக்சிடெண்டல் பிரைம் மினிஸ்டர்' - தேவகவுடா  == india
    ?“புல்லட் ரயில் இருக்கட்டும்.. இந்த ரயிலை கவனியுங்கள்” - பிரதமரை விமர்சித்த பாஜக முன்னாள் அமைச்சர் == tamilnadu

### Actual labels
    (torch) ~/projects/text_classification/main$ head ../dataset/label.txt 
    sports
    tamilnadu
    politics
    india
    sports
    india
    tamilnadu
    special-news
    india
    india
