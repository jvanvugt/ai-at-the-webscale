import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tqdm import *

from encoding import *
from models import *
from aiws import api
from login_info import USERNAME, PASSWORD
api.authenticate(USERNAME, PASSWORD)

REQUEST_NUMBERS = 10000

DATA_FOLDER = 'data'

def get_contexts(run_id):
    file_path = os.path.join(DATA_FOLDER, str(run_id)) + '.pkl'
    if os.path.isfile(file_path):
        return pd.read_pickle(file_path)
    else:
        print 'downloading data for run_id:', run_id
        ctxs = [api.get_context(run_id, i)['context'] for i in trange(REQUEST_NUMBERS)]
        df = pd.DataFrame(ctxs)
        df.to_pickle(file_path)
        return df

def plot_context_distributions(run_id):
    context = get_contexts(run_id)
    subplots = plt.subplots(2, 2)[1].ravel()
    for column, plot in zip(context.drop('visitor_id', 1), subplots):
        plot.set_title(column)
        context[column].value_counts().plot.pie(ax=plot)

    print 'Top 10 visitors'
    print context.visitor_id.value_counts().head(n=10)

    print 'Average number of visits:'
    print context.visitor_id.value_counts().mean()
    plt.show()

def test_all_options(run_id, option, action, action_name=None):
    file_path = os.path.join(DATA_FOLDER, '{}_{}'.format(run_id, option)) + '.pkl'

    if os.path.exists(file_path):
        return pd.read_pickle(file_path)

    if action_name is None:
        action_name = option

    contexts = get_contexts(run_id).to_dict('records')
    df = pd.DataFrame()
    for i, context in tqdm(enumerate(contexts)):
        for o in OPTIONS[option]:
            action[action_name] = o
            context['success'] = api.serve_page(run_id, i, **action)['success']
            context[action_name] = o
            df = df.append(context, ignore_index=True)

    df.to_pickle(file_path)
    return df

def test_actions():
    action = {
        'header': 5,
        'language': 'EN',
        'adtype': 'banner',
        'color': 'red',
        'price': 30.
    }
    df = test_all_options(1, 'color', action)
    df.color[df.success != 0.].value_counts().plot.bar()
    print df.color.head()
    print df.success.sum()
    plt.show()

def test_speed():
    contexts = get_contexts(0).to_dict('records')
    model = BootstrapModel(ContextlessThompsonModel,
        100, alpha=0.1, beta=0.1)
    for context in tqdm(contexts):
        context = encode_context(context)
        action = model.propose(context)
        model.update(context, action, True)


def run():
    test_actions()
    # plot_context_distributions(6)
    # test_speed()


if __name__ == '__main__':
    run()
