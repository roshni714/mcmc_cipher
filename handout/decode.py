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
	return ciphertext[0], transition

def get_inverse(f):
	dic = {}
	for i in f:
		dic[f[i]] = i
	return dic

def breakpoint_get_next_state(M, P, f1, f1_prime, f2, f2_prime, ciphertext, breakpoint):
		state1, transition1 = get_transitions(ciphertext[:breakpoint])
		state2, transition2 = get_transitions(ciphertext[breakpoint:])

		f1, out1 = get_next_state(M, P, f1, f1_prime, state1, transition1)
		f2, out2 = get_next_state(M, P, f2, f2_prime, state2, transition2)

		return f1, out1, f2, out2

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

def get_new_breakpoint(M, P, f1, f2, previous_breakpoint, ciphertext):
	f1_inverse = get_inverse(f1)
	f2_inverse = get_inverse(f2)
	  
	start1, transition1 = get_transitions(ciphertext[:previous_breakpoint])
	start2, transition2 = get_transitions(ciphertext[previous_breakpoint:])
	f_breakpoint = random.randint(1, len(ciphertext)-1)
#	f_breakpoint = previous_breakpoint + int((len(ciphertext) - previous_breakpoint)/2)
	f_start1, f_transition1 = get_transitions(ciphertext[:f_breakpoint])
	f_start2, f_transition2 = get_transitions(ciphertext[f_breakpoint:])
#	b_breakpoint = previous_breakpoint - int(previous_breakpoint/2)
#	b_start1, b_transition1 = get_transitions(ciphertext[:b_breakpoint])
#	b_start2, b_transition2 = get_transitions(ciphertext[b_breakpoint:])

	forward  =  posterior(M, P, f1_inverse, f_start1, f_transition1) + posterior(M, P, f2_inverse, f_start2, f_transition2)
#	backward = posterior(M, P, f1_inverse, b_start1, b_transition1) + posterior(M, P, f2_inverse, b_start2, b_transition2)
		
#	if forward > backward:
#		new_breakpoint = f_breakpoint
#		new_post = forward
#	else:
#		new_breakpoint = b_breakpoint
#		new_post = backward

	original = (posterior(M, P, f1_inverse, start1, transition1) + posterior(M, P, f2_inverse, start2, transition2))
	balance = forward - original

	acceptance_factor = min(0, balance)
	v = random.random()

	if v < math.exp(acceptance_factor):
			#accept
		return f_breakpoint, 0
	else:
			#reject
		return previous_breakpoint, 1


def breakpoint_metropolis_hastings(M, P, ciphertext):
	#initial state
	permutation1 = np.random.permutation(28)
	permutation2 = np.random.permutation(28)
	f1 = {}
	f2 = {}
	for i in range(len(ALPHABET)):
		f1[ALPHABET[i]] = ALPHABET[permutation1[i]]
		f2[ALPHABET[i]] = ALPHABET[permutation2[i]]
	f1_prime = get_proposal_distribution(f1)
	f2_prime = get_proposal_distribution(f2)
	breakpoint = int(len(ciphertext)/2)

	count1 = 0
	count2 = 0
	found_cipher1 = False
	found_cipher2 = False
	found_breakpoint = False
	break_count = 0
	for j in range(NUM_ITERATIONS):
		#get a proposal
		if (not found_cipher1) and (not found_cipher2):
			f1, out1, f2, out2 = breakpoint_get_next_state(M, P, f1, f1_prime, f2, f2_prime, ciphertext, breakpoint)
			breakpoint, out = get_new_breakpoint(M, P, f1, f2, breakpoint, ciphertext)
			f1_prime = get_proposal_distribution(f1)
			f2_prime = get_proposal_distribution(f2)

		elif not found_cipher1:
			state1, transition1 = get_transitions(ciphertext[:breakpoint])
			f1, out1 = get_next_state(M, P, f1, f1_prime, state1, transition1)
			breakpoint, out = get_new_breakpoint(M, P, f1, f2, breakpoint, ciphertext)
			f1_prime = get_proposal_distribution(f1)
		elif not found_cipher2:
			state2, transition2 = get_transitions(ciphertext[breakpoint:])
			f2, out2 = get_next_state(M, P, f2, f2_prime, state2, transition2) 
			breakpoint, out = get_new_breakpoint(M, P, f1, f2, breakpoint, ciphertext)
			f2_prime = get_proposal_distribution(f2)
		else:
			breakpoint, out = get_new_breakpoint(M, P, f1, f2, breakpoint, ciphertext)
		if count1 >= 1000:
			found_cipher1 = True
		if count2 >= 1000:
			found_cipher2 = True
		if break_count >= 5000:
			found_breakpoint = True
		if out1 == 0:
			count1 = 0
		else:
			count1 += 1
		if out2 == 0:
			count2 = 0
		else:
			count2 += 1
		if out == 0:
			break_count = 0
		else:
			break_count += 1

		#print("f1: {}, f2: {}, breakpoint: {}".format(count1, count2, break_count))
		#print("found_cipher1 = {}".format(found_cipher1))
		#print("found_cipher2 = {}".format(found_cipher2))
		#print("found_breakpoint = {}, breakpoint = {}".format(found_breakpoint, breakpoint))

		if found_cipher1 and found_cipher2 and found_breakpoint:
			break
			
	return f1, f2, breakpoint

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
	if has_breakpoint == False:
		f = metropolis_hastings(M, P, start, transitions)
		inverse_f = get_inverse(f)
		decoded = "".join([inverse_f[i] for i in ciphertext])

	else:
		f1, f2, breakpoint  = breakpoint_metropolis_hastings(M, P, ciphertext)
		inverse_f1 = get_inverse(f1)
		inverse_f2 = get_inverse(f2)
		decoded1 = "".join([inverse_f1[i] for i in ciphertext[:breakpoint]])
		decoded2 = "".join([inverse_f2[i] for i in ciphertext[breakpoint:]])
		decoded = decoded1 + decoded2
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

with open("test_ciphertext_breakpoint.txt") as f:
	ciphertext = f.read().rstrip()
	decode(ciphertext, True)
