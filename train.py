from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from time import time
import os

## SETTINGS ##

def ensure_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_file(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        data = f.read()
    return data

def preprocessing(data):
    ''' seperete sentences by '\n' and remove redudent spaces '''
    lines = data.split('\n')
    sentences = [[token for token in line.split(' ') if token != ''] for line in lines]
    return sentences

class loss_record(CallbackAny2Vec):
    '''Callback to record loss after each epoch.'''
    def __init__(self, loss_record = [], logging=False):
        self.epoch = 0
        self.logging= logging
        self.total_loss = 0
        self.record = loss_record

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss() - self.total_loss
        self.record.append(loss)
        self.total_loss = model.get_latest_training_loss()
        if self.logging:
            print('[ INFO ] Loss after epoch {:3d}: {:10.3f}'.format(self.epoch, loss))
        self.epoch += 1

class log_epoch(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.epoch_start_time = None

    def on_epoch_begin(self, model):
        print(f'[ INFO ] Epoch {self.epoch} start...')
        self.epoch_start_time = time()

    def on_epoch_end(self, model):
        train_time = time() - self.epoch_start_time
        print(f'[ INFO ] Epoch {self.epoch} end. Time eplapse: {train_time}')
        self.epoch += 1

def load_pretrain(model, pretrained_path):
    print('[ INFO ] loading pretrained vocabulary list...')
    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
    del pretrained_model   # free memory

    print('[ INFO ] loading pretrained model...')
    model.intersect_word2vec_format(pretrained_path, binary=True, lockf=0.0)    # lockf: freeze or not

    return model 

def train(input_file, output_dir, epoch, alpha, dim, window, eval_file, pretrained_model=None):
    # preprocessing
    print('[ INFO ] data processing...')
    training_data = load_file(input_file)
    training_data = preprocessing(training_data)

    # prepare model
    losses = []
    model = Word2Vec(size = dim, 
                     min_count = 1,
                     alpha = alpha,
                     window = window,
                     callbacks = [log_epoch(), loss_record(losses, True)])
    model.build_vocab(training_data)
    example_count = model.corpus_count

    # load pretrained
    if pretrained_model:
        model = load_pretrain(model, pretrained_model)

    # training
    print('[ INFO ] training start.')
    model.train(training_data,
                total_examples = example_count,
                epochs = epoch,
                compute_loss = True,
                callbacks = [log_epoch(), loss_record(losses, True)])
    # model = Word2Vec(training_data,
    #                   size = dim,
    #                   iter = epoch,
    #                   compute_loss = True,
    #                   callbacks=[log_epoch(), loss_record(losses, True)])
    print('[ INFO ] finished')

    # evaluating
    if eval_file:
        print('[ INFO ] evaluating...')
        result = model.wv.evaluate_word_analogies(eval_file)
        print(f'[ INFO ] evaluating finished. Accuracy = {result[0]}')

    # save model
    if output_dir:
        print(f'[ INFO ] saving data to {output_dir} ...')
        ensure_exist(output_dir)
        model.save(os.path.join(output_dir, 'model'))
        model.wv.save(os.path.join(output_dir, 'vecotr.kv'))
        with open(os.path.join(output_dir, 'loss'), 'w+') as f:
            for idx, loss in enumerate(losses):
                f.write(f"{idx}\t{loss}\n")
        if eval_file:
            with open(os.path.join(output_dir, 'accuracy'), 'w+') as f:
                f.write(f"{result[0]}\n")    # write accuracy
                f.write(str(result[1]))      # write evaluation log

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', type=str, required=True, help='training data')
        parser.add_argument('-o', '--output', type=str, help='folder to store output')
        parser.add_argument('-e', '--epoch', type=int, default=5,help='#iter')
        parser.add_argument('-d', '--dim', type=int, default=300, help='embedding dimension')
        parser.add_argument('-w', '--window', type=int, default=5, help='window size')
        parser.add_argument('-a', '--alpha', type=float, default=0.025, help='initial learning rate')
        parser.add_argument('-p', '--pretrained', nargs='?', help='pretrained model path')
        parser.add_argument('-eval', '--evaluate', type=str, help='evaluation data')
        return parser.parse_args()
    args = parse_args()

    print()
    print( 'Get Setting: ')
    print(f'  + input: {args.input}')
    print(f'  + output: {args.output}')
    print(f'  + alpha: {args.alpha}')
    print(f'  + epoch: {args.epoch}')
    print(f'  + dimension: {args.dim}')
    print(f'  + window size: {args.window}')
    print(f'  + evaluate file: {args.evaluate}')
    print( '  + pretrained model: {}'.format(args.pretrained if args.pretrained else 'None'))
    print()

    train(args.input, args.output, args.epoch, args.alpha, args.dim, args.window, args.evaluate, args.pretrained)
