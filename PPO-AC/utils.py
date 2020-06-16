from numpy.random import randint 

def loginto(path, msg):
    f = open(path, "a")
    f.write(msg)
    f.close()
    
def compute_gae(next_state, rewards, values, done_list, g_lambda, gamma):
    values = values + [next_state]
    gae = 0
    returns = []
    for idx in reversed(range(len(rewards))):
        terminal_coef = (1-done_list[idx])
        
        delta = rewards[idx] + gamma * \
            values[idx + 1] * terminal_coef - values[idx]
        
        gae = delta+g_lambda * gamma * terminal_coef * gae

        returns.insert(0, gae+values[idx])

    return returns

def normalize(x):
    x -= x.mean()
    x /= (x.std()+1e-8)
    return x

def batch_iter(states, actions, log_probs, returns, advantages, size):
    batch_size = states.size(0)

    for _ in range(batch_size // size):
        rand_ids = randint(0, batch_size, size)

        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

