import torch
import pandas as pd
import numpy as np

DIR_PATH = '/home/victorialena/ogb/examples/linkproppred/'

import pdb

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


class MultiLogger(object):
    def __init__(self, metrics, runs, info=None):
        self.loggers = {m: Logger(runs, info) for m in metrics}
        self.meta = {m: [[] for _ in range(runs)] for m in ['Loss', 'Epoch']}

    def __getitem__(self, arg):
        return self.loggers[arg]

    def keys(self):
        return self.loggers.keys()

    def add_results(self, run, epoch, loss, results):
        try:
            for key, result in results.items():
                self.loggers[key].add_result(run, result)
            self.meta['Loss'][run].append(loss)
            self.meta['Epoch'][run].append(epoch)
        except:
            AssertionError('Failed to save results to Logger.')

    def save_as(self, filename, dir=DIR_PATH+'data/'):
        data = {'Run': sum([[i]*len(_) for i,_ in enumerate(self.meta['Epoch'])], []),
                'Epoch': sum(self.meta['Epoch'], [])}
        for key, logger in self.loggers.items():
            X = np.stack(sum(logger.results, [])).T
            for x, label in zip(X, ['_train', '_valid', '_test']):
                data[key+label] = x
        stats = pd.DataFrame(data)
        stats.to_csv(dir+filename)