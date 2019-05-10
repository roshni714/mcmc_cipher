import numpy as np
import random
import math

NUM_ITERATIONS= 10000

def get_alphabet():
    with open('data/alphabet.csv') as f:
        a = f.read().rstrip().split(",")
    return a

ALPHABET = get_alphabet()
SIZE = len(ALPHABET)
LETTER_TO_INDEX = {ALPHABET[i]:i for i in range(len(ALPHABET))}

def get_transitions(ciphernums):
        new_transition = np.zeros((SIZE, SIZE))
        new_transition[ciphernums[1:], ciphernums[:-1]] += 1
	return ciphernums[0], new_transition

def get_inverse(f):
    new_f = np.zeros(SIZE, dtype=int)
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
		return f_prime, 0, new_ll 
	else:
			#reject
		return f, 1, old_ll
def metropolis_hastings(M, P, start, transitions):
	#initial state
	f= np.random.permutation(SIZE)

	f_prime = get_proposal_distribution(f)
	count = 0
        ll = -float("inf")
	for j in range(NUM_ITERATIONS):
		#get a proposal
		f, out, ll  = get_next_state(M, P, f, f_prime, start, transitions)
		if out == 0:
			count = 0
		else:
			count += out
		if count == 1000:
			break
		f_prime = get_proposal_distribution(f)
	return f, ll

def get_new_breakpoint(M, P, f1, f2, previous_breakpoint, ciphernums):
	f1_inverse = get_inverse(f1)
	f2_inverse = get_inverse(f2)
	  
	f_breakpoint = int(np.random.randn(1) *20 + previous_breakpoint) % len(ciphernums)

        while f_breakpoint == 0 or f_breakpoint == len(ciphernums)-1:
            f_breakpoint = int(np.random.randn(1) *20 + previous_breakpoint) % len(ciphernums)
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
	f1 = np.random.permutation(SIZE)
	f2 = np.random.permutation(SIZE)
	f1_prime = get_proposal_distribution(f1)
	f2_prime = get_proposal_distribution(f2)
	breakpoint = int(len(ciphernums)/2)

        best_iter = 0
        last_accepted_iter = 0

        best_f1 = f1
        best_f2 = f2
        best_b = breakpoint
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
	return best_f1, best_f2, best_b, ll_best_so_far

def get_proposal_distribution(f):
        new_f = np.copy(f)
	keys_to_swap = np.random.choice(f, 2, replace=False)

	first_map = f[keys_to_swap[0]]
	second_map = f[keys_to_swap[1]]

	new_f[keys_to_swap[0]] = second_map
	new_f[keys_to_swap[1]] = first_map

	return new_f

def posterior(M, P, f_inverse, start, transitions):
	probability = P[f_inverse[start]]
        i = np.arange(0, SIZE).repeat(SIZE)
        j = np.arange(0, SIZE).reshape(1, SIZE).repeat(SIZE, axis=0).flatten()
        p1 = np.sum(transitions[i, j] * M[f_inverse[i], f_inverse[j]])
	return probability+p1

def ciphertext_to_nums(ciphertext):
    ciphernum = np.array([LETTER_TO_INDEX[i] for i in ciphertext])
    return ciphernum

BREAKPOINT_ATTEMPTS = 10 
NO_BREAKPOINT_ATTEMPTS = 5 
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
                ciphernums = ciphertext_to_nums(ciphertext)
	        f_best = np.random.permutation(SIZE)
                ll_best_so_far = -float("inf")
                for i in range(NO_BREAKPOINT_ATTEMPTS):
                    start, transitions = get_transitions(ciphernums)
                    f, ll_b= metropolis_hastings(M, P, start, transitions)

                    if ll_b > ll_best_so_far:
                        f_best = f
                        ll_best_so_far = ll_b

	        inverse_f = get_inverse(f_best)
		decoded = "".join([ALPHABET[i] for i in inverse_f[ciphernums]])

	else:
                ciphernums = ciphertext_to_nums(ciphertext)
	        f1_best = np.random.permutation(SIZE)
                f2_best = np.random.permutation(SIZE)
                break_best = len(ciphernums)/2
                ll_best_so_far = -float("inf")
                for i in range(BREAKPOINT_ATTEMPTS):
                    f1, f2, breakpoint, ll_b  = breakpoint_metropolis_hastings(M, P, ciphernums)
                    if ll_b > ll_best_so_far:
                        f1_best = f1
                        f2_best = f2
                        break_best = breakpoint
                        ll_best_so_far = ll_b

		inverse_f1 = get_inverse(f1_best)
		inverse_f2 = get_inverse(f2_best)
		decoded1 = "".join([ALPHABET[i] for i in inverse_f1[ciphernums[:break_best]]])
		decoded2 = "".join([ALPHABET[i] for i in inverse_f2[ciphernums[break_best:]]])
		decoded = decoded1 + decoded2
	return decoded

