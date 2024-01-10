import pandas as pd
import numpy as np
import sklearn as sk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import unicodedata
import string
from datetime import datetime
from spellchecker import SpellChecker
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# TODOS:
# LSA fixen

# Read the CSV file
def read_file(file_path='./archive/reddit_opinion_climate_change.csv'):
    
    df = pd.read_csv(file_path)
    raw_comments = df['self_text'].values.astype(str)
    return raw_comments


#########################################################################################
####  PREPROCESSING  ####################################################################
#########################################################################################

def preprocessing(raw_comments, method='lemmatization'):
    
    # convert to lowercase
    prep_comments = np.char.lower(raw_comments)
  
    # remove URLs
    def remove_urls(text):
        url_pattern = re.compile(r'.*(http://|https://|www\.)\S+') # filters URLs in the form of [click](https://www.source.com) etc.
        return url_pattern.sub(r'', text)

    prep_comments = np.vectorize(remove_urls)(prep_comments)
    
    # apply a translation table which removes all punctuation
    punctuation_table = str.maketrans('', '', string.punctuation)
    prep_comments = np.char.translate(prep_comments, punctuation_table)

    # remove line breaks
    prep_comments = np.char.replace(prep_comments, '\n', ' ')

    # remove stop words and words with less than 3 characters
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['lol', 'yes', 'yep', 'due', 'sure', 'whatever', 'it', 'its', 'say', 'one', 'like', 'really', 'would', 'dont', 'hes', 'thats', 'youre', 'theyre', 'didn', 'wasn', 'wouldn', 'shouldn', 'couldnt', 'cant', 'wont', 'isnt', 'arent', 'didnt', 'doesnt', 'ive', 'youve', 'theyve', 'weve'])
    prep_comments = np.char.split(prep_comments)
    for i in range(len(prep_comments)):
        prep_comments[i] = [word for word in prep_comments[i] if word not in stop_words and len(word) > 2]
    prep_comments = [' '.join(comment) for comment in prep_comments]

    # remove numbers
    prep_comments = [re.sub(r'\d+', '', comment) for comment in prep_comments]
    # CO2 will remain as co

    # remove extra whitespace
    prep_comments = [re.sub(' +', ' ', comment) for comment in prep_comments]

    # remove accents
    prep_comments = [unicodedata.normalize('NFKD', comment).encode('ASCII', 'ignore').decode('utf-8', 'ignore') for comment in prep_comments]
    

    """
    # spell checking is not used, since it takes too long and does not improve the results significantly
    spell = SpellChecker(language='en')
    test_comments = prep_comments[1000:1010]
    test_comments = np.char.split(test_comments)
    for i in range(len(test_comments)):
        test_comments[i] = [spell.correction(word) if spell.correction(word) is not None else '' for word in test_comments[i]]
    test_comments = [' '.join(comment) for comment in test_comments]
    """
        
    # lemmatization OR stemming
    if method == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        nltk.download('wordnet') 

        prep_comments = np.char.split(prep_comments)
        for i in range(len(prep_comments)):
            prep_comments[i] = [lemmatizer.lemmatize(word) for word in prep_comments[i]]
        prep_comments = [' '.join(comment) for comment in prep_comments]
        
    elif method == 'stemming':
        stemmer = PorterStemmer()
        prep_comments = np.char.split(prep_comments)
        for i in range(len(prep_comments)):
            prep_comments[i] = [stemmer.stem(word) for word in prep_comments[i]]
        prep_comments = [' '.join(comment) for comment in prep_comments]
    
    # remove duplicate comments, which are for example copied through social media answer chains 
    seen = set()
    prep_comments = [comment for comment in prep_comments if not (comment in seen or seen.add(comment))]
    
    
    return prep_comments


#########################################################################################
####  VECTORIZING  ######################################################################
#########################################################################################

### Bag of Words with sklearn

def perform_bow(data):
   
    vectorizer_bow = CountVectorizer()
    bow = vectorizer_bow.fit_transform(data)
    terms_bow = vectorizer_bow.get_feature_names_out()
    # bow is a matrix which has the vocabulary as columns and the comments as rows
    # bow[i,j] is the number of times the word j appears in comment i
    # terms_bow is a list of the words in the vocabulary

    return bow, terms_bow


### TF-IDF with sklearn

def perform_tfidf(data):
    
    vectorizer_tfidf = TfidfVectorizer()
    tfidf = vectorizer_tfidf.fit_transform(data)
    terms_tfidf = vectorizer_tfidf.get_feature_names_out()
    # tfidf is a matrix which has the vocabulary as columns and the comments as rows
    # tfidf[i,j] is the tf-idf value of word j in comment i
    # terms_tfidf is a list of the words in the vocabulary
    
    return tfidf, terms_tfidf


#########################################################################################
####  THEMATIC ANALYSIS  ################################################################
#########################################################################################

### Latent Semantic Analysis (LSA) with sklearn

def perform_lsa(terms, matrix, themes):

    # Singular Value Decomposition (SVD) reduces the dimensionality of the tf-idf matrix
    # n_components is the number of topics to be found

    lsa = TruncatedSVD(n_components=themes)
    # lsa.components_ is a matrix which has the topics as rows 
    # lsa.components_[i,j] is the weight of word j in topic i

    # apply SVD to the vectorized matrix
    lsa.fit(matrix)

    # identify the most important words for each topic
    print("LSA Results: \n \n")
    
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f'./results/lsa_results_{now_str}.csv'
    with open({filename}, 'w') as f:
        for i, comp in enumerate(lsa.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
            print("Top 10 words for topic #"+str(i)+": ")
            for t in sorted_terms:
                print(t[0])
                f.write(t[0] + ', ')
            f.write('\n')
            print(" ")
    print("Results saved in ./results/lsa_results_$timestamp.csv \n \n")


    
### Latent Dirichlet Allocation (LDA) with sklearn

def perform_lda(terms, matrix, themes):

    # Latent Dirichlet Allocation (LDA) is a probabilistic topic model
    # n_components is the number of topics to be found, random_state is the seed for the random number generator
    # set random_state to 0 for reproducibility
    lda = LatentDirichletAllocation(n_components=themes, random_state=0)

    lda.fit(matrix)

    print("LDA Results: \n \n")
    
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f'./results/lda_results_{now_str}.csv'
    with open({filename}, 'w') as f:
        for i, topic in enumerate(lda.components_):
            print(f"Top 10 words for topic #{i}:")
            term_list = [terms[i] for i in topic.argsort()[-10:]]
            print(term_list)
            f.write(', '.join(term_list) + "\n")
            print("\n")
    print("Results saved in ./results/lda_results_$timestamp.csv \n \n")
    
#########################################################################################
####  MAIN  #############################################################################
#########################################################################################

def main():
    
    print("Welcome to the Comment Analysis Tool! \n \n This tool allows you to analyze datasets of comments in CSV format using different methods. \n Please enter the path to your CSV file, or press enter to work with an example dataset about reddit opinions on climate change: \n \n")
    user_file = input()
    
    if user_file:
        try:
            raw_comments = read_file(user_file)
            print("User file successfully loaded. \n")
        except:
            print("Invalid file path. Please enter a valid path.")
            main()
    else:
        try:
            raw_comments = read_file()
            print("Example dataset successfully loaded. \n")
        except:
            print("Example dataset not found. Please use a valid path.")
            main()

    print(raw_comments[900:950])

    print("Do you want to use Lemmatization or Stemming for Preprocessing? \n \n 1 = Lemmatization (recommended) \n 2 = Stemming (More context may be lost!) \n")
    try:
        word_preprocess = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
        
    if vector_method != 1 and vector_method != 2:
        print("Invalid input. Please enter 1 or 2.")
        main()
    
    print("Do you want to use Bag of Words or TF-IDF for vectorizing? \n \n 1 = Bag of Words \n 2 = TF-IDF \n")
    try:
        vector_method = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
    
        
    print("Do you want to use LSA or LDA for theme analysis? \n \n 1 = LSA \n 2 = LDA \n")
    try:
        analysis_method = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
        
    if analysis_method != 1 and analysis_method != 2:
        print("Invalid input. Please enter 1 or 2.")
        main()
    
    print("Please enter the number of themes you want to find: \n \n")
    try:
        themes = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
        

    print("Your dataset contains "+str(len(raw_comments))+" comments. \n \nPreprocessing...\n \n" )
    try: 
        if word_preprocess == 1:
            prep_comments = preprocessing(raw_comments)
        elif word_preprocess == 2:
            prep_comments = preprocessing(raw_comments, 'stemming')
    except:
        print( "Preprocessing failed. Please try again.")
        main()
    
    print(prep_comments[900:950])
    
    print("Preprocessing successful. \n \nVectorizing...")
    if vector_method == 1:
        try:
            matrix, terms = perform_bow(prep_comments)
        except:
            print("Vectorizing with BoW failed. Please try again.")
            main()
    elif vector_method == 2:
        try: 
            matrix, terms = perform_tfidf(prep_comments)
        except:
            print("Vectorizing with TF-IDF failed. Please try again.")
            main()
    

    print("Vectorizing done. \n \nAnalyzing for themes...\n \n" )
    if analysis_method == 1:
        try:
            perform_lsa(terms, matrix, themes)
        except:
            print("Analysis with LSA failed. Please try again.")
            main()
    elif analysis_method == 2:
        try:
            perform_lda(terms, matrix, themes)
        except:
            print("Analysis with LDA failed. Please try again.")
            main()
    

    print("Analysis done. \n \n Do you want to perform another analysis? \n \n 1 = Yes \n 2 = No \n")
    try:
        repeat = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        print("Goodbye!")
    
    if repeat == 1:
        main()
    else:
        print("Goodbye!")

# call main function
main()