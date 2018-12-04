import csv
import matplotlib.pyplot as plt

def load_data_0():
    data, labels = [], []
    with open('deceptive-opinion-2.csv', newline = '') as fp:
        reader = csv.reader(fp, delimiter = ',')
        for line in reader:
            validity = line[0]
            review = line[4].strip()
            data.append(review)
            labels.append(validity)
    return data[1:], labels[1:]

def load_data():
    data, labels = [], []
    with open('machine_data-3.csv', newline = '') as fp:
        reader = csv.reader(fp, delimiter = ',')
        for line in reader:
            validity = line[0]
            review = line[1].strip()
            data.append(review)
            labels.append(validity)
    return data, labels

def get_words_from_review(review):
    words = review.split()
    real_words = []
    for w in words:
        t = w.strip("',.?!()")
        if len(t) > 0:
            real_words.append(t)
    return real_words

def get_sentences_without_markers_from_review(review):
    r1 = [x for x in review.split('.') if len(x) > 0]
    r2 = []
    for i in r1:
        t1 = [y for y in i.split('?') if len(y) > 0]
        for j in t1:
            r2.append(j)
    r3 = []
    for k in r2:
        t2 = [z for z in k.split('!') if len(z) > 0]
        for l in t2:
            r3.append(l)
    return r3

def get_sentence_count(review):
    return len(get_sentences_without_markers_from_review(review))

def get_word_count(review):
    return len(get_words_from_review(review))

def get_number_count(review):
    words = get_words_from_review(review)
    number_count = len(words)
    for w in words:
        try:
            float(w)
        except:
            number_count -= 1
    return number_count

def get_character_count(review):
    return len(review)

def get_average_sentence_length(review):
    sentences = get_sentences_without_markers_from_review(review)
    total_sentence_length = 0
    for s in sentences:
        total_sentence_length += len(s)
    return total_sentence_length / len(sentences)

def get_average_word_length(review):
    words = get_words_from_review(review)
    total_word_length = 0
    for w in words:
        total_word_length += len(w)
    return total_word_length / len(words)

def get_unique_word_percentage(review):
    words = get_words_from_review(review)
    unique_words = set()
    for w in words:
        unique_words.add(w)
    return len(unique_words) / len(words)

def print_average(X, y, callback, description):
    count_of_t = 0
    count_of_f = 0
    feature_sum_of_t = 0
    feature_sum_of_f = 0
    for i in range(len(X)):
        if y[i] == "truthful":
            count_of_t += 1
            feature_sum_of_t += callback(X[i])
        else:
            count_of_f += 1
            feature_sum_of_f += callback(X[i])
    print(description + ' (truthful):', feature_sum_of_t / count_of_t)
    print(description + ' (deceptive):', feature_sum_of_f / count_of_f)
    plt.cla()
    plt.title(description)
    plt.bar(range(2), [feature_sum_of_t / count_of_t, feature_sum_of_f / count_of_f], tick_label = ["truthful", "deceptive"])
    plt.savefig(description)

def first_measure():
    print("HAND WRITTEN vs REAL:")
    X0, y0 = load_data_0()
    print_average(X0, y0, get_sentence_count, "Average Sentence Count")
    print_average(X0, y0, get_word_count, "Average Word Count")
    print_average(X0, y0, get_number_count, "Average Number Count")
    print_average(X0, y0, get_character_count, "Average Character Count")
    #print_average(X0, y0, get_average_sentence_length, "average average sentence length")
    #print_average(X0, y0, get_average_word_length, "average average word length")
    print_average(X0, y0, get_unique_word_percentage, "Average Unique Word Percentage")

def second_measure():
    print("MACHINE GENERATED vs REAL:")
    X, y = load_data()
    indices = []
    for i in range(len(X)):
        if len(get_words_from_review(X[i])) == 0 or len(get_sentences_without_markers_from_review(X[i])) == 0:
            indices.append(i)
    for ind in indices:
        X.pop(ind)
        y.pop(ind)
    print_average(X, y, get_sentence_count, "average sentence count")
    print_average(X, y, get_word_count, "average word count")
    print_average(X, y, get_number_count, "average number count")
    print_average(X, y, get_character_count, "average character count")
    #print_average(X, y, get_average_sentence_length, "average average sentence length")
    #print_average(X, y, get_average_word_length, "average average word length")
    print_average(X, y, get_unique_word_percentage, "average unique word percentage")

if __name__ == '__main__':
    first_measure()
    second_measure()
