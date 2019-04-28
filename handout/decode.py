import numpy as np
import string
import pandas as pd 
import random

NUM_ITERATIONS= 3000

LETTER_TO_INDEX = {'y': 24, 'e': 4, 'l': 11, 'f': 5, '.': 26, 'c': 2, 's': 18, 'h': 7, 'r': 17, 'x': 23, 'n': 13, 'p': 15, 'z': 25, 'q': 16, 'b': 1, 'g': 6, 'o': 14, 't': 19, 'v': 21, 'u': 20, 'j': 9, 'd': 3, 'a': 0, 'i': 8, 'w': 22, 'm': 12, ' ': 27, 'k': 10}

INDEX_TO_LETTER = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: '.', 27: ' '}

ALPHABET = string.ascii_lowercase + ". "
def make_dictionary():
	dic = {}
	count = 0
	for letter in string.ascii_lowercase:
		dic[count] = letter
		count +=1

	dic[26] = "."
	dic[27] = " "
	return dic

def get_inverse(f):
	dic = {}
	for i in f:
		dic[f[i]] = i
	return dic

def get_next_state(M, P, f, f_prime, ciphertext):

	f_inverse = get_inverse(f)
	f_prime_inverse = get_inverse(f)
	balance = posterior(M, P, f_inverse, ciphertext)/posterior(M, P, f_prime_inverse, ciphertext)

	acceptance_factor = min(1, balance)

	v = random.uniform(0, 1)

	if v < acceptance_factor:
		return f_prime

	else:
		return f
def metropolis_hastings(M, P, ciphertext):

	#initial state
	permutation = np.random.permutation(28)
	f = {}
	for i in range(len(ALPHABET)):
		f[ALPHABET[i]] = ALPHABET[permutation[i]]

	f_prime = get_proposal_distribution(f)
	for i in range(NUM_ITERATIONS):
		#get a proposal
		f = get_next_state(M, P, f, f_prime, ciphertext)
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

def posterior(M, P, f_inverse, ciphertext):

	probability = P[f_inverse[ciphertext[0]]]

	for i in range(len(ciphertext)-1):
		probability *= M[f_inverse[ciphertext[i+1]]][f_inverse[ciphertext[i]]]

	return probability

def decode(ciphertext, has_breakpoint):
	M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",")
	P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",")

	f = metropolis_hastings(M, P, ciphertext)

	decoded = "".join([f[i] for i in ciphertext])
	
	return decoded