from __future__ import division

import sys

from threading import Thread

import numpy as np
from tqdm import *
# import matplotlib.pyplot as plt

from aiws import api
from models import *
from encoding import encode_context, decode_action
from login_info import USERNAME, PASSWORD

api.authenticate(USERNAME, PASSWORD)

REQUEST_NUMBERS = 10000

def run_single_id(run_id, show_progress=True):
    range_func = trange if show_progress else xrange
    print 'starting run_id: ', run_id
    reward = 0
    successes = 0
    model = ContextualThompsonModel(alpha=.1, beta=.1)
    # mean_reward = np.zeros(REQUEST_NUMBERS / 100)
    for rn in range_func(REQUEST_NUMBERS):
        # if rn % 100 == 0:
        #     mean_reward[rn / 100] = reward / (rn + 1e-9)
        context = api.get_context(run_id=run_id, request_number=rn)['context']
        context = encode_context(context)
        action = model.propose(context)
        decoded_action = decode_action(action)
        result = api.serve_page(run_id=run_id, request_number=rn, **decoded_action)
        reward += decoded_action['price'] * result['success']
        if result['success']:
            successes += 1
        model.update(context, action, result['success'])
    # plt.plot(mean_reward)
    # plt.show()
    mean_reward = reward / REQUEST_NUMBERS
    print 'Mean reward for run_id', run_id, ':', mean_reward
    print 'Successes for run_id', run_id, ':' , successes
    # print model.successes
    return mean_reward

def run(id=0):
    return run_single_id(id)

def validate():
    api.reset_leaderboard()
    for i in xrange(5000, 5010):
        thread = Thread(target=run_single_id, args=(i, False))
        thread.start()


if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate()
    elif '--rid' in sys.argv:
        run(int(sys.argv[sys.argv.index('--rid') + 1]))
    elif '--train' in sys.argv:
        mean_reward = np.mean([run(id) for id in xrange(100, 120)])
        print 'mean reward over 20 runs: ', mean_reward
    else:
        run()
