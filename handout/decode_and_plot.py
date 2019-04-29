import numpy as np
import string
import random
import math
import plt

NUM_ITERATIONS= 10000

LETTER_TO_INDEX = {'y': 24, 'e': 4, 'l': 11, 'f': 5, ' ': 26, 'c': 2, 's': 18, 'h': 7, 'r': 17, 'x': 23, 'n': 13, 'p': 15, 'z': 25, 'q': 16, 'b': 1, 'g': 6, 'o': 14, 't': 19, 'v': 21, 'u': 20, 'j': 9, 'd': 3, 'a': 0, 'i': 8, 'w': 22, 'm': 12, '.': 27, 'k': 10}

INDEX_TO_LETTER = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ' ', 27: '.'}

ALPHABET = string.ascii_lowercase + " ."

def get_transitions(ciphertext):
    transition = np.zeros((28, 28))
    for i in range(len(ciphertext)-1):
        first = ciphertext[i]
        second = ciphertext[i+1]
        transition[LETTER_TO_INDEX[second]][LETTER_TO_INDEX[first]]+=1
    return ciphertext[0], transition

def get_inverse(f):
	dic = {}
	for i in f:
		dic[f[i]] = i
	return dic

def get_next_state(M, P, f, f_prime, start, transitions):
	f_inverse = get_inverse(f)
	f_prime_inverse = get_inverse(f_prime)
	balance = posterior(M, P, f_prime_inverse, start, transitions)-posterior(M, P, f_inverse, start, transitions)
	acceptance_factor = min(0, balance)
	v = random.random()

	if v < math.exp(acceptance_factor):
            #accept
		return f_prime, 0
	else:
            #reject
		return f, 1
def metropolis_hastings(M, P, start, transitions):

	#initial state
	permutation = np.random.permutation(28)
	f = {}
	for i in range(len(ALPHABET)):
		f[ALPHABET[i]] = ALPHABET[permutation[i]]

	f_prime = get_proposal_distribution(f)
        count = 0
	for j in range(NUM_ITERATIONS):
		#get a proposal
		f, out = get_next_state(M, P, f, f_prime, start, transitions)
                if out == 0:
                    count = 0
                else:
                    count += out
                if count == 1000:
                    break
		f_prime = get_proposal_distribution(f)
	return f



def get_proposal_distribution(f):
	new_f = {}
	for i in f:
		new_f[i] = f[i]

	keys_to_swap = random.sample(f.keys(), 2)

	first_map = f[keys_to_swap[0]]
	second_map = f[keys_to_swap[1]]

	new_f[keys_to_swap[0]] = second_map
	new_f[keys_to_swap[1]] = first_map
	return new_f

def posterior(M, P, f_inverse, start, transitions):
	probability = math.log(P[LETTER_TO_INDEX[f_inverse[start]]])
	for i in range(len(transitions)):
                for j in range(len(transitions[0])):
                        if transitions[i][j] != 0:
                            probability += transitions[i][j] * math.log(M[LETTER_TO_INDEX[f_inverse[INDEX_TO_LETTER[i]]]][LETTER_TO_INDEX[f_inverse[INDEX_TO_LETTER[j]]]])

	return probability

def decode(ciphertext, has_breakpoint):
	M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
	P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")
        start, transitions = get_transitions(ciphertext)

        for i in range(len(M)):
            for j in range(len(M)):
                if M[i][j] == 0:
                    M[i][j] += math.exp(-20)
                M[i][j] = float(M[i][j])
            if P[i] == 0:
                P[i] += math.exp(-20)
                P[i] = float(P[i])
        f = metropolis_hastings(M, P, start, transitions)
        inverse_f = get_inverse(f)
	decoded = "".join([inverse_f[i] for i in ciphertext])
        return decoded

def verify_proposal():
    perm = np.random.permutation(28)
    dic = {}
    for i in range(len(ALPHABET)):
        dic[ALPHABET[i]] = INDEX_TO_LETTER[perm[i]]

    f = get_proposal_distribution(dic)
    count = 0
    for i in f:
        if f[i] != dic[i]:
            count += 1
    assert count ==2
def verify_inv():
    perm = np.random.permutation(28)
    dic = {}
    for i in range(len(ALPHABET)):
        dic[ALPHABET[i]] = INDEX_TO_LETTER[perm[i]]

    f = get_inverse(dic)
    for i in f:
        if dic[f[i]] != i:
            assert False
