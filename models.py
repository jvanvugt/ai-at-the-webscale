import itertools

import numpy as np

from encoding import *

class RandomModel(object):
    """
    Picks an action at random
    """
    def propose(self, context):
        action = np.zeros(ACTION_VECTOR_LENGTH)
        action[-1] = np.random.normal(15, 5)
        return action

    def update(self, context, action, success):
        pass

class ConstantModel(object):
    """
    Always picks the aciton provided in the constructor
    """
    def __init__(self, action):
        self.action = action

    def propose(self, context):
        return self.action

    def update(self, context, action, success):
        pass

class EnsembleModel(object):
    """
    An ensemble of models.
    Picks the action with the that's chosen by most models 
    """
    def __init__(self, model, n, **kwargs):
        """
        model: class of the model that should be used
        n: number of models in the ensemble
        kwargs: arguments passed to the constructor of model
        """
        self.models = [model(**kwargs) for _ in xrange(n)]

    def propose(self, context):
        actions = np.sum([model.propose(context) for model in self.models], axis=0)
        action = np.zeros(ACTION_VECTOR_LENGTH)
        i = 0
        header_action_idx = np.argmax(actions[i:i + len(HEADER)]) + i
        action[header_action_idx] = 1
        i += len(HEADER)
        adtype_action_idx = np.argmax(actions[i:i + len(ADTYPE)]) + i
        action[adtype_action_idx] = 1
        i += len(ADTYPE)
        color_action_idx = np.argmax(actions[i:i + len(COLOR)]) + i
        action[color_action_idx] = 1
        i += len(COLOR)
        language_action_idx = np.argmax(actions[i:i + len(LANGUAGE_ACT)]) + i
        action[language_action_idx] = 1
        i += len(LANGUAGE_ACT)
        action[i] = actions[-1] // len(self.models)
        return action

    def update(self, context, action, success):
        for model in self.models:
            model.update(context, action, success)

class ContextlessThompsonModel(object):
    """
    Thompson sampling (sampling from a beta distribution) that does not keep
    the context account in any way.

    This version does not have any arms for the interaction between the elements of the action
    Thus, it assumes no interaction between the actions
    """
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

        # Compute the expected values
        scaled_probs = np.array(probs) * self.prices
        action = np.zeros(len(probs))
        action[np.argmax(scaled_probs)] = 1
        return action

class InteractionContextlessThompsonModel(object):
    """
    Thompson sampling with arms for the interaction between arms.
    Does not take context into account.
    """
    def __init__(self, alpha=1., beta=1.):
        self.prices = np.arange(5., 50., 5.)
        self.n = 0
        self.successes = np.zeros(len(HEADER) * len(ADTYPE) * len(COLOR) * len(LANGUAGE_ACT))
        self.price_successes = np.zeros(len(self.prices))
        self.alpha = alpha
        self.beta = beta
        self.actions = list(itertools.product(HEADER, ADTYPE, COLOR, LANGUAGE_ACT))
    
    def propose(self, context):
        probs = [np.random.beta(s + 1, self.n - s + 1) for s in self.successes]
        action_index = np.argmax(probs)
        action_names = self.actions[action_index]
        action = np.zeros(ACTION_VECTOR_LENGTH)
        i = 0
        action[i + HEADER.index(action_names[0])] = 1
        i += len(HEADER)
        action[i + ADTYPE.index(action_names[1])] = 1
        i += len(ADTYPE)
        action[i + COLOR.index(action_names[2])] = 1
        i += len(COLOR)
        action[i + LANGUAGE_ACT.index(action_names[3])] = 1
        i += len(LANGUAGE_ACT)
        price_probs = [np.random.beta(s + 1, self.n - s + 1) for s in self.price_successes]
        # Compute the expected value
        price_probs *= self.prices
        action[i] = self.prices[np.argmax(price_probs)] 
        return action
    
    def update(self, context, action, success):
        self.n += self.beta
        if success:
            action = decode_action(action)
            action_names = (action['header'], action['adtype'], action['color'], action['language'])
            action_index = self.actions.index(action_names)
            self.successes[action_index] += self.alpha
            self.price_successes[self.prices == action['price']] += self.alpha



class ContextualThompsonModel(object):
    """
    Thompson sampling without arms for interaction between actions.
    Does have arms for interaction with context.
    Does not take interaction between context into account
    """
    def __init__(self, alpha=1., beta=1.):
        self.prices = np.arange(5., 50., 5.)
        self.n = 0
        context_length = len(AGENT) + len(OS) + len(LANGUAGE_CTX) + len(REFERRER)
        self.successes = np.zeros((ACTION_VECTOR_LENGTH - 1 + len(self.prices), context_length))
        self.alpha = alpha
        self.beta = beta

    def propose(self, context):
        # Disregard age and visitor_id
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
        # Disregard age and visitor_id
        context = context[2:]
        self.n += self.beta
        if success:
            price_idx = np.where(self.prices == action[-1])
            prices = np.zeros(len(self.prices))
            prices[price_idx] = 1.
            action = np.concatenate((action[:-1], prices))
            for a, c in itertools.product(np.flatnonzero(action), np.flatnonzero(context)):
                self.successes[a, c] += self.alpha

    def pick_action(self, start_index, end_index, context):
        """
        Pick an action according to the beta distribution

        - start_index: The first index of a specific action
                       in the action vector
        - end_index: The first element of the next action

        - context: The encoded context
        """
        action_indices = xrange(start_index, end_index)
        context_indices = np.flatnonzero(context)

        # Find the number of successes per action possibility
        # and context
        successes = [self.successes[a, c] for a, c in itertools.product(action_indices, context_indices)]
        probs = [np.random.beta(s + 1, self.n - s + 1) for s in successes]
        action = np.zeros(end_index - start_index)

        # Use integer division instead of modulo because itertools.product
        # iterates over the last argument first
        action[np.argmax(probs) / len(context_indices)] = 1
        return action

    def pick_price(self, start_index, end_index, context):
        """
        Very similar to pick_action, but scales the probabilities by
        the price, so we choose a price based on the expected value
        """
        n_options = end_index - start_index
        action_indices = xrange(start_index, end_index)
        context_indices = np.flatnonzero(context)
        successes = [self.successes[a, c] for a, c in itertools.product(action_indices, context_indices)]
        probs = np.array([np.random.beta(s + 1, self.n - s + 1) for s in successes])

        # Compute the expected values
        scaled_probs = (probs.reshape(n_options, -1) * self.prices.reshape(-1, 1)).reshape(len(successes))
        action = np.zeros(n_options)
        action[np.argmax(scaled_probs) / len(context_indices)] = 1
        return action

class BootstrapModel(object):
    """
    Implementation of Bootstrap (Thompson) sampling: https://arxiv.org/abs/1410.4009
    """
    def __init__(self, model=ContextualThompsonModel, n=10, **kwargs):
        self.models = [model(**kwargs) for _ in range(n)]

    def propose(self, context):
        return np.random.choice(self.models).propose(context)

    def update(self, context, action, success):
        for model in self.models:
            if np.random.rand(1) > 0.5:
                model.update(context, action, success)
