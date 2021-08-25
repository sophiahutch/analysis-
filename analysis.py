import os # used to locate the txt files for processing
import nltk # used to stem processing
import sys
import time
import operator # used to sort the lists for top 20 terms
import math # required for log() function

def remove_non_alpha(word):
    front_done = False
    back_done = False
    while (not front_done):
        if len(word)<2:
            return word
        if (not word[0].isalpha()):
            word = word[1:]
        if word[0].isalpha():
            front_done = True
    
    while (not back_done):
        if len(word)<2:
            return word
        if (not word[-1].isalpha()):
            word = word[:-1]
        if word[-1].isalpha():
            back_done = True
    return word


path = 'txt_files/'

files = os.listdir(path)

# Calculate the IDF for all terms found in first document
N = 0
for name in files:
    if name.endswith(".txt"):
        N+=1

print(f"Processing {N} research papers.")

# define the maximum length of a term to consider
MAX_TERM_LENGTH = 30
# create a stemmer object
stemmer = nltk.wordnet.WordNetLemmatizer()

# master dictionary
master = {}
start = time.time()
doc_id = 1
for filename in files:
    if filename.endswith(".txt"):
        print (f"Processing document {doc_id} of {N}")
        doc_id +=1
        fullname = path + filename
    else:
        continue # skip any files that do not end with .txt
    
    # open each file and print the filename
    f = open (fullname,encoding='utf-8')

    # obtain all of the words as a string
    lines = f.readlines()
    for data in lines:
        
        # setup current dictionary for the current file
        current = {}
        
        word_list = nltk.word_tokenize(data)
        for word in word_list:
            word = remove_non_alpha(word)
            
            # if word is long enough lemmatize and update master
            if (len(word)>1 and len(word)<=MAX_TERM_LENGTH):
                word = stemmer.lemmatize(word)
                if (not word in current):
                    current[word] = 1
                    if (not word in master):
                        master[word] = 1
                    else:
                        master[word] = 1 + master[word]
    f.close()

end = time.time()
master_time = round((end-start)/60,1)
print (f"The master dictionary has been created. It took {master_time} minutes")

# determine the tdf for each file
for filename in files:
    if filename.endswith(".txt"):
        fullname = path + filename
        
        # open each file and print the filename
        f = open (fullname,encoding='utf-8')
        print (f"Calculating TDF for {fullname}")
    else:
        continue # skip any files not ending with .txt
    
    word_freq = {}
    
    #obtain all of the words as a string
    lines = f.readlines()
    word_count=0
    for data in lines:
        
        # create dictionary of words for the file
        word_list = nltk.word_tokenize(data)
        partial_word_count = len(word_list)
        word_count +=partial_word_count # keep track of total words in doc
        
        for word in word_list:
            word = remove_non_alpha(word)
            
            # if word is long enough lemmatize and update word_freq for document
            if (len(word)>1 and len(word)<=MAX_TERM_LENGTH):
                word = stemmer.lemmatize(word)
                if (not word in word_freq):
                    word_freq[word] = 1
                else:
                    word_freq[word] = 1 + word_freq[word]
    
    f.close()

    # create new dictionary of tdf - term frequency density
    tdf = {}
    for key in word_freq.keys():
        tdf[key] = float(word_freq[key])/word_count
    
    sorted_tdf = sorted(tdf.items(), reverse=True, key=operator.itemgetter(1))
   # sorted_counts = sorted(word_freq.items(), reverse=True, key=operator.itemgetter(1))

    top_20_filename = fullname.split(".")[0]+".top"
    file_out = open(top_20_filename,"w",encoding="utf-8")
    file_out.write("Top 20 key terms\n")
    
    # Write out the top 20 frequent terms or less
    
    for i in range(len(sorted_tdf)):
        file_out.write(f'{sorted_tdf[i][0]}\n')
        if i==19:
            break
    file_out.close()

    tf_idf = []
    for item in sorted_tdf:
        term = item[0]
        term_count = master[term]
        # print (f'{term} : {term_count} : {tdf[term]}')
        tf_idf.append([term,math.log(N/term_count) * float(tdf[term])])
                      
    sorted_tf_idf = sorted(tf_idf, reverse=True, key=operator.itemgetter(1))

    # create important term file
    td_idf_filename = fullname.split(".")[0]+".tfidf"
    file_out = open(td_idf_filename,"w",encoding="utf-8")
    file_out.write("Top 20 important terms\n")

    # writing td-idf to a file
    for i in range(len(sorted_tf_idf)):
        file_out.write(f'{sorted_tf_idf[i][0]}\n')
        if i==19:
            break
    file_out.close()