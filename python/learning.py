"""
Created on Feb 27, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import sys
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Local import
import config
import metadata


def learn_prior(paths):
    def normalize_prob(cpt):
        for s in range(cpt.shape[0]):
            cpt[s, :] = cpt[s, :]/np.sum(cpt[s, :])

        return cpt

    if not os.path.exists(os.path.join(paths.tmp_root, 'activity_corpus.p')):
        sys.exit('Ground truth pickle file not found.')
    activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))

    prior_folder = os.path.join(paths.tmp_root, 'prior')
    if not os.path.exists(prior_folder):
        os.mkdir(prior_folder)

    # TODO: modify the prior
    action_cpt = np.ones((len(metadata.subactivities), len(metadata.actions))) * 0.3
    affordance_cpt = np.ones((len(metadata.subactivities), len(metadata.affordances))) * 0.1
    object_cpt = np.ones((len(metadata.subactivities), len(metadata.objects))) * 0.0001
    object_affordance_cpt = np.ones((len(metadata.objects), len(metadata.affordances))) * 0.0001
    duration_dict = dict()
    for s in metadata.subactivities:
        duration_dict[s] = list()

    for activity in activity_corpus:
        print activity
        for tpg in activity_corpus[activity]:
            for t in tpg.terminals:
                s = t.subactivity
                duration_dict[s].append(t.end_frame - t.start_frame)
                duration_dict['prior'].append(t.end_frame - t.start_frame)

                a = t.subactivity
                action_cpt[metadata.subactivity_index[s], metadata.action_index[a]] += 1
                action_cpt[metadata.subactivity_index['prior'], metadata.action_index[a]] += 1
                for u in t.affordance:
                    affordance_cpt[metadata.subactivity_index[s], metadata.affordance_index[u]] += 1
                    affordance_cpt[metadata.subactivity_index['prior'], metadata.affordance_index[u]] += 1
                for io, o in enumerate(t.objects):
                    object_cpt[metadata.subactivity_index[s], metadata.object_index[o]] += 1
                    object_cpt[metadata.subactivity_index['prior'], metadata.object_index[o]] += 1
                    object_affordance_cpt[metadata.object_index[o], metadata.affordance_index[t.affordance[io]]] += 1

    object_affordance_cpt[:, -1] = 0
    object_affordance_cpt[:, -1] = np.max(object_affordance_cpt, axis=1)

    action_cpt = normalize_prob(action_cpt)
    affordance_cpt = normalize_prob(affordance_cpt)
    object_cpt = normalize_prob(object_cpt)
    object_affordance_cpt = normalize_prob(object_affordance_cpt)
    with open(os.path.join(prior_folder, 'action_cpt.json'), 'w') as output_file:
        json.dump(action_cpt.tolist(), output_file, indent=4, separators=(',', ': '))
    with open(os.path.join(prior_folder, 'affordance_cpt.json'), 'w') as output_file:
        json.dump(affordance_cpt.tolist(), output_file, indent=4, separators=(',', ': '))
    with open(os.path.join(prior_folder, 'object_cpt.json'), 'w') as output_file:
        json.dump(object_cpt.tolist(), output_file, indent=4, separators=(',', ': '))
    with open(os.path.join(prior_folder, 'object_affordance_cpt.json'), 'w') as output_file:
        json.dump(object_affordance_cpt.tolist(), output_file, indent=4, separators=(',', ': '))

    plt_idx = 0
    duration_prior = dict()
    for s, durations in duration_dict.items():
        mu, std = scipy.stats.norm.fit(durations)
        duration_prior[s] = [mu, std]

    #     plt_idx += 1
    #     ax = plt.subplot(2, 6, plt_idx)
    #
    #     plt.hist(durations, normed=True)
    #     dmin, dmax = plt.xlim()
    #     x = np.linspace(dmin, dmax, 100)
    #     p = scipy.stats.norm.pdf(x, mu, std)
    #     plt.plot(x, p, 'k', linewidth=2)
    #
    #     ax.set_title(s)
    # plt.show()
    with open(os.path.join(prior_folder, 'duration_prior.json'), 'w') as output_file:
        json.dump(duration_prior, output_file, indent=4, separators=(',', ': '))


def main():
    paths = config.Paths()
    learn_prior(paths)


if __name__ == '__main__':
    main()
