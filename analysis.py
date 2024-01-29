import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import string
from datetime import datetime
from gensim.models import CoherenceModel, Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances
from spellchecker import SpellChecker


#########################################################################################
####  INPUT  ############################################################################
#########################################################################################

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
    
    # remove emails
    def remove_emails(text):
        email_pattern = re.compile(r'\S+@\S+')
        return email_pattern.sub(r'', text)

    prep_comments = np.vectorize(remove_emails)(prep_comments)
    
    # apply a translation table which removes all punctuation
    punctuation_table = str.maketrans('', '', string.punctuation)
    prep_comments = np.char.translate(prep_comments, punctuation_table)

    # remove line breaks
    prep_comments = np.char.replace(prep_comments, '\n', ' ')

    # remove special characters
    prep_comments = [re.sub(r'[^a-zA-Z\s]', '', comment) for comment in prep_comments]
    
    # remove extra whitespace
    prep_comments = [re.sub(' +', ' ', comment) for comment in prep_comments]
    
    # remove stop words and words with less than 3 characters
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['ampxb', 'lol', 'yes', 'yep', 'due', 'sure', 'whatever', 'it', 'its', 'say', 'one', 'like', 'really', 'would', 'dont', 'hes', 'thats', 'youre', 'theyre', 'didn', 'wasn', 'wouldn', 'shouldn', 'couldnt', 'cant', 'wont', 'isnt', 'arent', 'didnt', 'doesnt', 'ive', 'youve', 'theyve', 'weve'])
    prep_comments = np.char.split(prep_comments)
    for i in range(len(prep_comments)):
        prep_comments[i] = [word for word in prep_comments[i] if word not in stop_words and len(word) > 2]
    prep_comments = [' '.join(comment) for comment in prep_comments]

    """
    # spell checking is not used, since it takes too long and does not improve the results significantly
    
    spell = SpellChecker(language='en')
    comments = prep_comments
    comments = np.char.split(comments)
    for i in range(len(comments)):
        comments[i] = [spell.correction(word) if spell.correction(word) is not None else '' for word in comments[i]]
    comments = [' '.join(comment) for comment in comments]
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
    
    # remove duplicate comments, which are e.g. copied through social media answer chains 
    seen = set()
    prep_comments = [comment for comment in prep_comments if not (comment in seen or seen.add(comment))]
    
    
    return prep_comments


#########################################################################################
####  VECTORIZING  ######################################################################
#########################################################################################

### Bag of Words with sklearn

def perform_bow(data, ngram_range=(1,1)):
   
    vectorizer_bow = CountVectorizer(ngram_range=ngram_range)
    bow = vectorizer_bow.fit_transform(data)
    terms_bow = vectorizer_bow.get_feature_names_out()
    # bow is a matrix which has the vocabulary as columns and the comments as rows
    # bow[i,j] is the number of times the word j appears in comment i
    # terms_bow is a list of the words in the vocabulary

    return bow, terms_bow


### TF-IDF with sklearn

def perform_tfidf(data, ngram_range=(1,1)):
    
    vectorizer_tfidf = TfidfVectorizer(ngram_range=ngram_range)
    tfidf = vectorizer_tfidf.fit_transform(data)
    terms_tfidf = vectorizer_tfidf.get_feature_names_out()
    # tfidf is a matrix which has the vocabulary as columns and the comments as rows
    # tfidf[i,j] is the tf-idf value of word j in comment i
    # terms_tfidf is a list of the words in the vocabulary
    
    return tfidf, terms_tfidf


### Word2Vec with gensim

def perform_word2vec(data, min_count=1, size=100, window=5):
    """
    :param data: list of tokenized texts (a list of lists of words)
    :param min_count: minimum number of occurrences of a word to be included in the model
    :param size: number of dimensions of the embedding vector --> size of hidden layer, higher value = higher accuracy but higher computational cost
    :param window: maximum distance between the current and predicted word within a sentence.
    :return: The trained Word2Vec model.
    """
    # Prepare data for Word2Vec (tokenization)
    data = [comment.split() for comment in data]

    # Train Word2Vec model
    w2v_model = Word2Vec(data, min_count=min_count, vector_size=size, window=window)

    return w2v_model



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
    now_str = now.strftime("%Y%m%d_%H%M")
    filename = f'./results/lsa_results_{now_str}.csv'
    topics = []  
    with open(filename, 'w') as f:
        for i, comp in enumerate(lsa.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
            print("Top 10 words for topic #"+str(i)+": ")
            topic_words = []
            for t in sorted_terms:
                print(t[0])
                f.write(t[0] + ', ')
                topic_words.append(t[0])  
            topics.append(topic_words)  
            f.write('\n')
            print(" ")
    print(f'Results saved in {filename} \n \n')
    
    return topics


    
### Latent Dirichlet Allocation (LDA) with sklearn

def perform_lda(terms, matrix, themes):

    # Latent Dirichlet Allocation (LDA) is a probabilistic topic model
    # n_components is the number of topics to be found, random_state is the seed for the random number generator
    # set random_state to 0 for reproducibility
    lda = LatentDirichletAllocation(n_components=themes, random_state=0)

    lda.fit(matrix)

    print("LDA Results: \n \n")
    
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M")
    filename = f'./results/lda_results_{now_str}.csv'
    topics = []  
    with open(filename, 'w') as f:
        for i, topic in enumerate(lda.components_):
            print(f"Top 10 words for topic #{i}:")
            term_list = [terms[i] for i in topic.argsort()[-10:]]
            print(term_list)
            f.write(', '.join(term_list) + "\n")
            topics.append(term_list)  
            print("\n")
    print(f'Results saved in {filename} \n \n')
    
    return topics



### Clustering with K-Means

def calculate_optimal_k(word_vectors, max_k=10):
    
    # calculate the optimal number of clusters using the elbow method
    # SSE = sum of squared errors (the distance of each point to its cluster center)
    # the optimal number of clusters is the point where the SSE stops decreasing significantly, so that adding another cluster does not improve the model
    print("Calculating optimal number of clusters using the elbow method... \n")
    
    sse = []
    list_k = list(range(2, max_k+1))

    # for each k, Clustering is performed and the SSE is added to the list
    for k in list_k:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(word_vectors)
        sse.append(km.inertia_)

    # comparing the differences between the SSEs of each k to find the elbow point
    diffs = np.diff(sse)
    diff_ratios = diffs[:-1] / diffs[1:]
    k_idx = np.argmax(diff_ratios) + 1
    optimal_k = list_k[k_idx]
    
    # visualization of elbow plot
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.plot(optimal_k, sse[k_idx], 'ro')  # mark the elbow point
    plt.text(optimal_k, sse[k_idx], '   Optimal k = '+str(optimal_k))
    
    # save elbow plot
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M")
    plot_elbow_filename = f'./results/elbow_plot_{now_str}.png'
    plt.savefig(plot_elbow_filename)
    print(f'Elbow plot saved in {plot_elbow_filename} \n \n The optimal number of clusters is {optimal_k}. \n \n')
    plt.close()
    
    return optimal_k


def perform_kmeans_clustering(model, max_themes=10):
    
    # K-Means Clustering for themes
    word_vectors = model.wv.vectors
    optimal_k = calculate_optimal_k(word_vectors, max_themes) 
    kmeans = KMeans(n_clusters=optimal_k) 
    clusters = kmeans.fit_predict(word_vectors)

    # visualization of clusters
    # reduce dimensionality of word vectors to 2D for visualization
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_vectors)
    
    print("K-Means Clustering Results: \n ")
    plt.scatter(result[:, 0], result[:, 1], c=clusters)
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M")
    plot_cluster_filename = f'./results/cluster_plot_{now_str}.png'
    plt.savefig(plot_cluster_filename)
    print(f'Plot saved in {plot_cluster_filename} \n \n')
    plt.close()
    
    # extract the most important words for each cluster and save it as the result
    filename = f'./results/cluster_results_{now_str}.csv'
    topics = [ ]
    with open(filename, 'w') as f:
        for i in range(optimal_k): 
            words_in_cluster = [word for word, cluster in zip(model.wv.index_to_key, clusters) if cluster == i]
            if len(words_in_cluster) > 10:
                # get the centroid of the current cluster (the mean of all word vectors in the cluster)
                centroid = kmeans.cluster_centers_[i]
                # calculate the distance of each word in the cluster to the centroid
                distances = pairwise_distances([centroid], model.wv[words_in_cluster])[0]
                # get the indices of the 5 words that are closest to the centroid
                top_indices = np.argsort(distances)[:10]
                # select only the top X words
                words_in_cluster = [words_in_cluster[index] for index in top_indices]
            print(f"Words in topic {i}: {words_in_cluster}")
            f.write(', '.join(words_in_cluster) + "\n")
            topics.append(words_in_cluster)
    print(f'Results saved in {filename} \n \n')
    
    return topics



### Coherence Score

def get_coherence(topics, prep_comments):
  
  # transform comments into list of lists of strings
  texts = [comment.split() for comment in prep_comments]
     
  # format topics to list of list of strings (separate n-grams by comma)
  split_grams = [[gram.split() for gram in topic] for topic in topics]
  fmt_topics = [ ]
  for topic in split_grams:
    fmt_topic = [ ]
    for gram in topic:
      fmt_topic.extend(gram)
    fmt_topics.append(fmt_topic)

  # create a gensim dictionary from comments
  dictionary = corpora.Dictionary(texts)
  
  # calculate coherence score with gensim model
  coherence_model = CoherenceModel(topics=fmt_topics, texts=texts, dictionary=dictionary, coherence='c_v')
  coherence = round(coherence_model.get_coherence(), 4)

  print(f'Coherence Score for resulting themes: {coherence} \n \n')







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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    print("Do you want to use Lemmatization or Stemming for Preprocessing? \n \n 1 = Lemmatization (recommended) \n 2 = Stemming (More context may be lost!) \n")
    try:
        word_preprocess = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
        
    if word_preprocess != 1 and word_preprocess != 2:
        print("Invalid input. Please enter 1 or 2.")
        main()
    
    print("Do you want to use Bag of Words, TF-IDF or Word2Vec for vectorizing? \n \n 1 = Bag of Words \n 2 = TF-IDF \n 3 = Word2Vec \n")
    try:
        vector_method = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
    
    if vector_method != 3: 
        print("Do you want to use LSA or LDA for theme analysis? \n \n 1 = LSA \n 2 = LDA \n")
        try:
            analysis_method = int(input())
        except ValueError:
            print("Invalid input. Please enter a number.")
            main()
            
        if analysis_method != 1 and analysis_method != 2:
            print("Invalid input. Please enter 1 or 2.")
            main()
    elif vector_method == 3:
        analysis_method = 3
        
    if analysis_method != 3:
        print("What is the size of n-grams you want to use for vectorizing? Beware that n-grams can increase complexity.  \n \n 1 = Unigrams \n 2 = Bigrams \n 3 = Trigrams \n ... \n n = N-Grams \n")
        try:
            n_gram_size = int(input())
        except ValueError:
            print("Invalid input. Please enter a number.")
            main()
            
    print("Please enter the number of themes you want to find: \n \n")
    try:
        themes = int(input())
    except ValueError:
        print("Invalid input. Please enter a number.")
        main()
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Your dataset contains "+str(len(raw_comments))+" comments. \n \nPreprocessing...\n \n" )
    
    try: 
        if word_preprocess == 1:
            prep_comments = preprocessing(raw_comments)
        elif word_preprocess == 2:
            prep_comments = preprocessing(raw_comments, 'stemming')
    except:
        print( "Preprocessing failed. Please try again.")
        main()
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    print("Preprocessing successful. \n \nVectorizing...")
    if vector_method == 1:
        try:
            matrix, terms = perform_bow(prep_comments, (n_gram_size, n_gram_size))
        except:
            print("Vectorizing with BoW failed. Please try again.")
            main()
    elif vector_method == 2:
        try: 
            matrix, terms = perform_tfidf(prep_comments, (n_gram_size, n_gram_size))
        except:
            print("Vectorizing with TF-IDF failed. Please try again.")
            main()
    elif vector_method == 3:
        try: 
            w2v_model = perform_word2vec(prep_comments)
        except:
            print("Vectorizing with TF-IDF failed. Please try again.")
            main()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Vectorizing done. \n \nAnalyzing for themes...\n \n" )
    if analysis_method == 1:
        try:
            topics = perform_lsa(terms, matrix, themes)
        except:
            print("Analysis with LSA failed. Please try again.")
            main()
    elif analysis_method == 2:
        try:
            topics = perform_lda(terms, matrix, themes, prep_comments)
        except:
            print("Analysis with LDA failed. Please try again.")
            main()
    elif analysis_method == 3:
        print("Since you chose Word2Vec for vectorizing, theme analysis is carried out using clustering.  \n \n")
        try:
            topics = perform_kmeans_clustering(w2v_model, themes)  
        except:
            print("Analysis with Clustering failed. Please try again.")
            main()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    print("Theme analysis done. Calculating coherence score for your themes... \n \n")
    get_coherence(topics, prep_comments)

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
if __name__ == "__main__":
    main()