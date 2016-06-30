import numpy as np

AGENT = ['Mozilla Firefox', 'Google Chrome', 'Safari', 'Opera', 'Internet Explorer']
OS = ['Windows', 'OSX', 'Android', 'iOS', 'Linux']
LANGUAGE_CTX = ['NL', 'EN', 'GE', 'Other']
REFERRER = ['Google', 'Bing', 'Other']

HEADER = [5, 15, 35]
ADTYPE = ['skyscraper', 'square', 'banner']
COLOR = ['green', 'blue', 'red', 'white'] # black never works
LANGUAGE_ACT = ['NL', 'EN', 'GE']

PRICE_MIN = 0.01
PRICE_MAX = 50.

ACTION_VECTOR_LENGTH = len(HEADER)+len(ADTYPE)+len(COLOR)+len(LANGUAGE_ACT)+1

OPTIONS = {
    'agent': AGENT, 'os': OS, 'language_ctx': LANGUAGE_CTX, 'referrer': REFERRER,
    'header': HEADER, 'adtype': ADTYPE, 'color': COLOR, 'language_act': LANGUAGE_ACT
}

def encode_context(context):
    id = np.array([context['visitor_id']])
    age = np.array([context['age']])
    agent = one_hot_encode('agent', context['agent'])
    os = one_hot_encode('os', context['os'])
    language = one_hot_encode('language_ctx', context['language'])
    referrer = one_hot_encode('referrer', context['referrer'])
    return np.concatenate((id, age, agent, os, language, referrer))

def decode_action(action):
    i = 0
    header = one_hot_decode('header', action[i:i+len(HEADER)])
    i += len(HEADER)
    adtype = one_hot_decode('adtype', action[i:i+len(ADTYPE)])
    i += len(ADTYPE)
    color = one_hot_decode('color', action[i:i+len(COLOR)])
    i += len(COLOR)
    language = one_hot_decode('language_act', action[i:i+len(LANGUAGE_ACT)])
    i += len(LANGUAGE_ACT)
    price = np.clip(action[i], PRICE_MIN, PRICE_MAX)
    return {
        'header': header,
        'adtype': adtype,
        'color': color,
        'price': price,
        'language': language
    }


def one_hot_decode(key, vector):
    one_idx = np.flatnonzero(vector)
    if len(one_idx) == 0:
        one_idx = np.random.randint(0, len(vector))
    elif len(one_idx) == 1:
        one_idx = one_idx[0]
    else:
        one_idx = np.random.choice(one_idx)
    return OPTIONS[key][one_idx]

def one_hot_encode(key, value):
    options = OPTIONS[key]
    vector = np.zeros(len(options))
    vector[options.index(value)] = 1
    return vector

if __name__ == '__main__':
    context =  {
        'visitor_id': 74,
        'agent': 'Mozilla Firefox',
        'os': 'Android',
        'language': 'NL',
        'age': 21,
        'referrer': 'Bing'
    }
    print encode_context(context)
