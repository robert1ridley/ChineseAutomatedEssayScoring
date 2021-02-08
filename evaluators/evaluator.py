from metrics.metrics import *
from utils import rescale_scores


class Evaluator():
    def __init__(self, dev_titles, dev_essays, dev_scores,
                 test_titles, test_essays, test_scores):
        self.dev_titles, self.test_titles = dev_titles, test_titles
        self.dev_essays, self.test_essays = dev_essays, test_essays
        self.dev_scores, self.test_scores = dev_scores, test_scores
        self.rescaled_dev_scores, self.rescaled_test_scores = rescale_scores(dev_scores), rescale_scores(test_scores)
        self.best_dev_qwk = -1
        self.best_test_qwk = -1


    def calc_kappa(self, dev_pred, test_pred, weight='quadratic'):
        self.dev_qwk = kappa(self.rescaled_dev_scores, dev_pred, weight)
        self.test_qwk = kappa(self.rescaled_test_scores, test_pred, weight)


    def evaluate(self, model, epoch, print_info=True):
        dev_pred = model.predict(self.dev_essays, batch_size=32).squeeze()
        test_pred = model.predict(self.test_essays, batch_size=32).squeeze()

        dev_pred_int = rescale_scores(dev_pred)
        test_pred_int = rescale_scores(test_pred)

        self.calc_kappa(dev_pred_int, test_pred_int)

        if self.dev_qwk > self.best_dev_qwk:
            self.best_dev_qwk = self.dev_qwk
            self.best_test_qwk = self.test_qwk
            self.best_dev_epoch = epoch
        if print_info:
            self.print_info()


    def print_info(self):
        print('[DEV] QWK: %.3f || BEST QWK: %.3f, BEST EPOCH: %.3f' %
              (self.dev_qwk, self.best_dev_qwk, self.best_dev_epoch))
        print('[TEST] QWK: %.3f || BEST QWK: %.3f, BEST EPOCH: %.3f' %
              (self.test_qwk, self.best_test_qwk, self.best_dev_epoch))
        print(
            '--------------------------------------------------------------------------------------------------------------------------')


    def print_final_info(self):
        print('[DEV] BEST QWK: %.3f, BEST EPOCH: %.3f' %
              (self.best_dev_qwk, self.best_dev_epoch))
        print('[TEST] BEST QWK: %.3f, BEST EPOCH: %.3f' %
              (self.best_test_qwk, self.best_dev_epoch))
        print(
            '--------------------------------------------------------------------------------------------------------------------------')