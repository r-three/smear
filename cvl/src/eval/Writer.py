class Writer(object):

    def __init__(self, file):
        self.write_file = open(file, 'w+')

    def add_batch(self, batch_input, batch_true_lbl, batch_pred_lbl):
        for (input, true_lbl, pred_lbl) in zip(batch_input, batch_true_lbl, batch_pred_lbl):
            line = '\t'.join([input, str(true_lbl), str(pred_lbl)]) + '\n'
            self.write_file.write(line)