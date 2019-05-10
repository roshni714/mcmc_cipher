import numpy as np
import string
import random
import math

NUM_ITERATIONS= 30000

LETTER_TO_INDEX = {'y': 24, 'e': 4, 'l': 11, 'f': 5, ' ': 26, 'c': 2, 's': 18, 'h': 7, 'r': 17, 'x': 23, 'n': 13, 'p': 15, 'z': 25, 'q': 16, 'b': 1, 'g': 6, 'o': 14, 't': 19, 'v': 21, 'u': 20, 'j': 9, 'd': 3, 'a': 0, 'i': 8, 'w': 22, 'm': 12, '.': 27, 'k': 10}

INDEX_TO_LETTER = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ' ', 27: '.'}

ALPHABET = string.ascii_lowercase + " ."

def get_transitions(ciphertext):
	transition = np.zeros((28, 28))
	for i in range(len(ciphertext)-1):
		first = ciphertext[i]
		second = ciphertext[i+1]
		transition[LETTER_TO_INDEX[second]][LETTER_TO_INDEX[first]]+=1
        assert(ciphertext[0] == ALPHABET[LETTER_TO_INDEX[ciphertext[0]]])
	return LETTER_TO_INDEX[ciphertext[0]], transition

def get_inverse(f):
    new_f = np.zeros(28, dtype=int)
    for i in range(len(f)):
        new_f[f[i]] = i
    return new_f

def breakpoint_get_next_state(M, P, f1, f1_prime, f2, f2_prime, ciphernums, breakpoint):
                f1, out1 = breakpoint_direct_posterior_next_state(M, P, f1, f1_prime, ciphernums[:breakpoint])
                f2, out2 = breakpoint_direct_posterior_next_state(M, P, f2, f2_prime, ciphernums[breakpoint:])
		return f1, out1, f2, out2

def direct_posterior(M, P, f_inverse, ciphernums):
    probability = P[f_inverse[ciphernums[0]]]
    probability +=  np.sum(M[f_inverse[ciphernums[1:]], f_inverse[ciphernums[:-1]]])
    return probability

def breakpoint_direct_posterior_next_state(M, P, f, f_prime, ciphernums):
	f_inverse = get_inverse(f)
	f_prime_inverse = get_inverse(f_prime)
	new_ll= direct_posterior(M, P, f_prime_inverse, ciphernums)
        old_ll= direct_posterior(M, P, f_inverse, ciphernums)
	acceptance_factor = min(0, new_ll-old_ll)
	v = random.random()

	if v < math.exp(acceptance_factor):
			#accept
		return f_prime, 0 
	else:
			#reject
		return f, 1 

def get_next_state(M, P, f, f_prime, start, transitions):
	f_inverse = get_inverse(f)
	f_prime_inverse = get_inverse(f_prime)
	new_ll= posterior(M, P, f_prime_inverse, start, transitions)
        old_ll= posterior(M, P, f_inverse, start, transitions)
	acceptance_factor = min(0, new_ll-old_ll)
	v = random.random()

	if v < math.exp(acceptance_factor):
			#accept
		return f_prime, 0 
	else:
			#reject
		return f, 1 
def metropolis_hastings(M, P, start, transitions):
	#initial state
	f= np.random.permutation(28)

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

def get_new_breakpoint(M, P, f1, f2, previous_breakpoint, ciphernums):
	f1_inverse = get_inverse(f1)
	f2_inverse = get_inverse(f2)
	  
	f_breakpoint = random.randint(1, len(ciphernums)-1)

        forward  =  direct_posterior(M, P, f1_inverse, ciphernums[:f_breakpoint]) + direct_posterior(M, P, f2_inverse, ciphernums[f_breakpoint:])
        original = (direct_posterior(M, P, f1_inverse, ciphernums[:previous_breakpoint]) + direct_posterior(M, P, f2_inverse, ciphernums[previous_breakpoint:]))
	balance = forward - original

	acceptance_factor = min(0, balance)
	v = random.random()

	if v < math.exp(acceptance_factor):
			#accept
		return f_breakpoint, 0, forward 
	else:
			#reject
		return previous_breakpoint, 1, original


def breakpoint_metropolis_hastings(M, P, ciphernums):
	#initial state
	f1 = np.random.permutation(28)
	f2 = np.random.permutation(28)
	f1_prime = get_proposal_distribution(f1)
	f2_prime = get_proposal_distribution(f2)
	breakpoint = int(len(ciphernums)/2)

        best_iter = 0
        last_accepted_iter = 0

        best_f1 = f1
        best_f2 = f2
        best_breakpoint = breakpoint
        ll_best_so_far = -float("inf")  
	for j in range(NUM_ITERATIONS):
		#get a proposal
                if(j- last_accepted_iter) <= 1500 and (j - best_iter) <= 5000:
			f1, rejected1, f2, rejected2 = breakpoint_get_next_state(M, P, f1, f1_prime, f2, f2_prime, ciphernums, breakpoint)
			breakpoint, rejectedb, llb= get_new_breakpoint(M, P, f1, f2, breakpoint, ciphernums)
			f1_prime = get_proposal_distribution(f1)
			f2_prime = get_proposal_distribution(f2)

                        if (not rejected1) or (not rejected2) or (not rejectedb):
                            last_accepted_iter = j

                        if llb > ll_best_so_far:
                            ll_best_so_far = llb
                            best_f1 = f1
                            best_f2 = f2
                            best_b = breakpoint
                            best_iter = j
                else:
                    break
	return f1, f2, breakpoint

def get_proposal_distribution(f):
	new_f = np.zeros(28, dtype=int)

        new_f = [f[i] for i in range(len(f))]

	keys_to_swap = random.sample(f, 2)

	first_map = f[keys_to_swap[0]]
	second_map = f[keys_to_swap[1]]

	new_f[keys_to_swap[0]] = second_map
	new_f[keys_to_swap[1]] = first_map

        verify_proposal(f, new_f)
	return new_f

def verify_proposal(f, new_f):
    count = 0
    for i in range(len(f)):
        if f[i] != new_f[i]:
            count +=1
    assert(count==2)


def posterior(M, P, f_inverse, start, transitions):
	probability = P[f_inverse[start]]
	for i in range(len(transitions)):
				for j in range(len(transitions[0])):
						if transitions[i][j] != 0:
							probability += transitions[i][j] * M[f_inverse[i]][f_inverse[j]]

	return probability

def ciphertext_to_nums(ciphertext):
    ciphernum = np.array([LETTER_TO_INDEX[i] for i in ciphertext])
    return ciphernum

def decode(ciphertext, has_breakpoint):
	M = np.loadtxt(open("data/letter_transition_matrix.csv", "rb"), delimiter=",", dtype=float)
	P = np.loadtxt(open("data/letter_probabilities.csv", "rb"), delimiter=",", dtype=float)

	for i in range(len(M)):
		for j in range(len(M)):
			if M[i][j] == 0:
				M[i][j] += math.exp(-20)
                        M[i][j] = math.log(M[i][j])
		if P[i] == 0:
			P[i] += math.exp(-20)
                P[i] = math.log(P[i])
	if has_breakpoint == False:
  	        start, transitions = get_transitions(ciphertext)
                f = metropolis_hastings(M, P, start, transitions)
		inverse_f = get_inverse(f)
		decoded = "".join([ALPHABET[inverse_f[LETTER_TO_INDEX[i]]] for i in ciphertext])

	else:
                ciphernums = ciphertext_to_nums(ciphertext)
		f1, f2, breakpoint  = breakpoint_metropolis_hastings(M, P, ciphernums)
		inverse_f1 = get_inverse(f1)
		inverse_f2 = get_inverse(f2)
		decoded1 = "".join([ALPHABET[i] for i in inverse_f1[ciphernums[:breakpoint]]])
		decoded2 = "".join([ALPHABET[i] for i in inverse_f2[ciphernums[breakpoint:]]])
		decoded = decoded1 + decoded2
	return decoded
"""
with open("test_ciphertext_breakpoint.txt") as f:
    ciphertext = f.read().rstrip()
    decoded = decode(ciphertext, True)
    print(decoded)
with open("test_plaintext.txt") as f2:
    plaintext = f2.read().rstrip()
    count = 0.
    for i in range(len(plaintext)):
        if plaintext[i] == decoded[i]:
            count +=1

    print("accuracy: {}".format(count/len(plaintext)))

"""
