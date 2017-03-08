# ------------------------------------------------------------------

#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedurce, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be ---
'''

data_list = sample_memo.strip().split()

words_to_guess = ["ahead", "could"]


def next_word_probability(sample_text, word):
    word_list = sample_text.split(" ")
    return_dict = {}
    previous_word = ""
    for nextWord in word_list:
        if previous_word == word:
            if nextWord in return_dict:
                return_dict[nextWord] += 1
            else:
                return_dict[nextWord] = 1
        previous_word = nextWord
    return return_dict


def LaterWords(sample, word, distance):
    '''@param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    '''

    # TODO: Given a word, collect the relative probabilities of possible following words
    # from @sample. You may want to import your code from the maximum likelihood exercise.
    next_word_dict = next_word_probability(sample, word)
    for i in range(1, distance):
        following_word = {}
        maxWord, maxValue = '', 0
        for k,v in next_word_dict.items():
            following_words = next_word_probability(sample, k)
            for next_k, next_v in following_words.items():
                next_v = next_v * v
                if next_k in following_word:
                    following_word[next_k] += next_v
                else:
                    following_word[next_k] = next_v
                if next_v > maxValue:
                    maxWord, maxValue = next_k, next_v

    return maxWord

print LaterWords(sample_memo, "and", 2)
