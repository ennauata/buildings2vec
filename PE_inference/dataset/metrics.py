import numpy as np

class Metrics(): 
    def __init__(self):
        self.curr_tp = 0.0
        self.curr_fp = 0.0
        self.n_samples = 0.0
        self.per_sample_score = {}

    def forward(self, im_id, gts, dets):
        
        per_sample_tp = 0.0
        per_sample_fp = 0.0
        found = [False] * gts.shape[0]
    
        for det in dets:

            # get closest gt
            near_gt = [0, 99999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt-det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt] 

            # hit (<= 8px) and not found yet 
            if near_gt[1] <= 8.0 and not found[near_gt[0]]:
                    per_sample_tp += 1.0
                    found[near_gt[0]] = True

            # not hit or already found
            else:
                per_sample_fp += 1.0

        # update counters
        self.curr_tp += per_sample_tp
        self.curr_fp += per_sample_fp
        self.n_samples += gts.shape[0]
        self.per_sample_score[im_id] = {'recall': per_sample_tp/gts.shape[0],
                                        'precision': per_sample_tp/(per_sample_tp+per_sample_fp+1e-8)} 
        return

    def calc_metrics(self):
        recall = self.curr_tp/self.n_samples
        precision = self.curr_tp/(self.curr_tp+self.curr_fp+1e-8)
        return recall, precision

    def print_metrics(self):

        # print recall

        recall = self.curr_tp/self.n_samples
        precision = self.curr_tp/(self.curr_tp+self.curr_fp+1e-8)

        print('All Samples\nrecall: %.3f\nprecision: %.3f\n' % (recall, precision))
        print('Per sample')
        for k in self.per_sample_score.keys():
            recall = self.per_sample_score[k]['recall']
            precision = self.per_sample_score[k]['precision']
            print('id: %s; recall: %.3f; precision: %.3f' % (k, recall, precision))
        return 

    def reset(self):
        self.curr_tp = 0.0
        self.curr_fp = 0.0
        self.n_samples = 0.0
        self.per_sample_score = {}