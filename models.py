import itertools

import numpy as np

from encoding import *

class RandomModel(object):

    def propose(self, context):
        action = np.zeros(ACTION_VECTOR_LENGTH)
        action[-1] = np.random.normal(15, 5)
        return action

    def update(self, context, action, success):
        pass

class ConstantModel(object):

    def __init__(self, action):
        self.action = action

    def propose(self, context):
        return self.action

    def update(self, context, action, success):
        pass


class ContextlessThompsonModel(object):

    def __init__(self, alpha=1., beta=1.):
        self.prices = np.arange(5., 50., 5.)
        self.successes = np.zeros(ACTION_VECTOR_LENGTH - 1 + len(self.prices))
        self.n = 0
        self.alpha = alpha
        self.beta = beta

    def propose(self, context):
        os_decoded = one_hot_decode('os', context[5:5+len(OS)])
        on_mobile = os_decoded == 'iOS' or os_decoded == 'Android'
        i = 0
        header = self.pick_action(i, i + len(HEADER))
        # if on_mobile:
        #     header = np.array([1, 0, 0])
        i += len(HEADER)
        adtype = self.pick_action(i, i + len(ADTYPE))
        # if on_mobile:
        #     adtype = np.array([1, 0, 0])
        i += len(ADTYPE)
        color = self.pick_action(i, i + len(COLOR))
        i += len(COLOR)
        language = self.pick_action(i, i + len(LANGUAGE_ACT))
        i += len(LANGUAGE_ACT)
        price_index = self.pick_price(i, i + len(self.prices))
        price = self.prices[np.nonzero(price_index)]
        action = np.concatenate((header, adtype, color, language, price))
        return action

    def update(self, context, action, success):
        context = context[2:]
        self.n += self.beta
        if success:
            price_idx = np.where(self.prices == action[-1])
            prices = np.zeros(len(self.prices))
            prices[price_idx] = 1.
            self.successes += np.concatenate((action[:-1], prices)) * self.alpha

    def pick_action(self, start_index, end_index):
        successes = self.successes[start_index:end_index]
        probs = [np.random.beta(s + 1, self.n - s + 1) for s in successes]
        action = np.zeros(len(probs))
        action[np.argmax(probs)] = 1
        return action

    def pick_price(self, start_index, end_index):
        successes = self.successes[start_index:end_index]
        probs = [np.random.beta(s + 1, self.n - s + 1) for s in successes]
        scaled_probs = np.array(probs) * self.prices
        action = np.zeros(len(probs))
        action[np.argmax(scaled_probs)] = 1
        return action

class ContextualThompsonModel(object):
    def __init__(self, alpha=1., beta=1.):
        self.prices = np.arange(5., 50., 5.)
        self.n = 0
        context_length = len(AGENT) + len(OS) + len(LANGUAGE_CTX) + len(REFERRER)
        self.successes = np.zeros((ACTION_VECTOR_LENGTH - 1 + len(self.prices), context_length))
        self.alpha = alpha
        self.beta = beta

    def propose(self, context):
        context = context[2:]
        i = 0
        header = self.pick_action(i, i + len(HEADER), context)
        i += len(HEADER)
        adtype = self.pick_action(i, i + len(ADTYPE), context)
        i += len(ADTYPE)
        color = self.pick_action(i, i + len(COLOR), context)
        i += len(COLOR)
        language = self.pick_action(i, i + len(LANGUAGE_ACT), context)
        i += len(LANGUAGE_ACT)
        price_index = self.pick_price(i, i + len(self.prices), context)
        price = self.prices[np.nonzero(price_index)]
        action = np.concatenate((header, adtype, color, language, price))
        return action


    def update(self, context, action, success):
        context = context[2:]
        self.n += self.beta
        if success:
            price_idx = np.where(self.prices == action[-1])
            prices = np.zeros(len(self.prices))
            prices[price_idx] = 1.
            action = np.concatenate((action[:-1], prices))
            for a, c in itertools.product(np.flatnonzero(action), np.flatnonzero(context)):
                self.successes[a, c] += self.alpha
        if self.n == 5000:
            print 'context:'
            print context
            print '\naction:'
            print action
            print '\nsuccesses'
            print self.successes

    def pick_action(self, start_index, end_index, context):
        successes = [self.successes[a, c] for a, c in zip(xrange(start_index, end_index), np.flatnonzero(context))]
        probs = [np.random.beta(s + 1, self.n - s + 1) for s in successes]
        action = np.zeros(end_index - start_index)
        action[np.argmax(probs) % len(action)] = 1
        return action

    def pick_price(self, start_index, end_index, context):
        n_options = end_index - start_index
        successes = [self.successes[a, c] for a, c in itertools.product(xrange(start_index, end_index), np.flatnonzero(context))]
        probs = np.array([np.random.beta(s + 1, self.n - s + 1) for s in successes])
        scaled_probs = (probs.reshape(n_options, -1) * self.prices.reshape(-1, 1)).reshape(len(successes))
        action = np.zeros(n_options)
        action[np.argmax(scaled_probs) % len(action)] = 1
        return action
