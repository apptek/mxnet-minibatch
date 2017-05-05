import numpy as np
import mxnet as mx
import random
from mxnet.io import DataIter, DataBatch
from mxnet import ndarray
import h5py
import bisect
import argparse

parser = argparse.ArgumentParser(description="Sequence sampling experiments on CHiME-4",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--output-file', type=str, default='out.cache',
                    help='posterior output cache file')
parser.add_argument('--input-file', type=str, default='data/valid.0001',
                    help='feature input file')
parser.add_argument('--num-layers', type=int, default=3,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=512,
                    help='hidden layer size')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='whether to use bidirectional layers')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=10,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=100,
                    help='show progress for every n batches')
# When training a deep, complex model, it's recommended to stack fused RNN cells (one
# layer per cell) together instead of one with all layers. The reason is that fused RNN
# cells doesn't set gradients to be ready until the computation for the entire layer is
# completed. Breaking a multi-layer fused RNN cell into several one-layer ones allows
# gradients to be processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')
parser.add_argument('--dropout', type=float, default='0.3',
                    help='dropout probability (1.0 - keep probability)')

parser.add_argument('--sampling', type=str, default=None,
                    help='sequence batch sampling method: random, sorted, #partitions '
                         'in sinusoidal sampling or comma separated list of buckets')


class UtteranceBatch(object):
  def __init__(self, data_names, data, label_names, label, bucket_key,
               utt_names=None, utt_lens=0):
    self.data = data
    self.label = label
    self.data_names = data_names
    self.label_names = label_names
    self.bucket_key = bucket_key
    self.utt_names = utt_names
    self.utt_lens = utt_lens

  @property
  def provide_data(self):
    return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

  @property
  def provide_label(self):
    if len(self.label_names):
      return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
    else:
      return None

class UtteranceIter(DataIter):
  """A data iterator for acoustic modeling of frame-wise labeled data.
  The iterator supports bucketing based on predefined bucket sizes,
  sorted and random sequence sampling, and sinusoidal sampling.  

  Parameters
  ----------
  utterances : list of list of list of floats
      utterance feature vectors with #sequences as major
  batch_size : int
      number of sequences per batch
  sampling : str, int, or list of int
      either 'sorted' or 'random' 
      or number of bins in sinusoidal sampling 
      or size of data buckets (automatically generated if None).
  """

  def __init__(self, utterances, states, names, batch_size, batching, data_name='data', label_name='labels', shuffle=True):
    super(UtteranceIter, self).__init__()
    minpad = 100
    sampling = range(minpad,max([len(s) for s in utterances]),minpad)
    self.max_len = max([len(utt) for utt in utterances])
    if sampling[-1] < self.max_len:
        sampling.append(self.max_len)

    self.idx = []
    if isinstance(batching, list):
      batching.sort()
      if batching[-1] < self.max_len:
        batching.append(self.max_len)
      sampling = batching

      self.data = [[] for _ in sampling]
      self.labels = [[] for _ in sampling]
      self.names = [[] for _ in sampling]
      self.lengths = [[] for _ in sampling]
      for utt, lab, name in zip(utterances, states, names):
        buck = bisect.bisect_left(sampling, len(utt))
        xin = np.full((sampling[buck],len(utt[0])), -1, dtype='float32')
        n_in = len(utt[0])
        xin[:len(utt)] = utt
        yout = np.full((sampling[buck],), -1, dtype='float32')
        yout[:len(utt)] = lab
        self.data[buck].append(xin)
        self.labels[buck].append(yout)
        self.names[buck].append(name)
        self.lengths[buck].append(len(utt))
      for i, buck in enumerate(self.data):
        self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        if len(buck) % batch_size != 0:
          self.idx.append((i,len(buck)-len(buck)%batch_size)) #[(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])

    else:
      self.idx.extend([[0,j] for j in range(0, len(utterances) - batch_size + 1, batch_size)])

    self.data = [np.asarray(i, dtype='float32') for i in self.data] # BTD
    self.labels = [np.asarray(i, dtype='float32') for i in self.labels] # BT
    self.curr_idx = 0

    self.batch_size = batch_size
    self.sampling = sampling
    self.batching = batching
    self.nddata = []
    self.ndlabel = []
    self.data_name = data_name
    self.label_name = label_name
    self.sampling = sampling
    self.default_key = max(sampling)
    self.shuffle = shuffle

    # we assume time major layout
    self.provide_data = [(self.data_name, (self.default_key, batch_size, n_in))]
    self.provide_label = [(self.label_name, (self.default_key, batch_size))]

    self.reset()

  def reset(self):
    self.curr_idx = 0
    self.nddata = []
    self.ndlabel = []

    if isinstance(self.batching, list):
      if self.shuffle:
        random.shuffle(self.idx) # shuffle bucket index
        for buck_utt, buck_lab in zip(self.data,self.labels): # shuffle sequence index within bucket
          rng_state = np.random.get_state()
          np.random.shuffle(buck_utt)
          np.random.set_state(rng_state)
          np.random.shuffle(buck_lab)

      for buck_utt,buck_lab in zip(self.data,self.labels):
        self.nddata.append(ndarray.array(buck_utt, dtype='float32'))
        self.ndlabel.append(ndarray.array(buck_lab, dtype='float32'))
    elif self.batching == 'sorted':
      self.idx.sort(key = lambda x:x[1])
      

  def next(self):
    if self.curr_idx == len(self.idx):
      raise StopIteration

    if isinstance(self.sampling, list):
      i, j = self.idx[self.curr_idx]

      slice_end = min(self.nddata[i].shape[0],j+self.batch_size)
      #slice_end = j + self.batch_size

      data = self.nddata[i][j:slice_end]
      label = self.ndlabel[i][j:slice_end]
      names = self.names[i][j:slice_end]
      lens = self.lengths[i][j:slice_end]
      data = ndarray.swapaxes(data, 0, 1) # TBD
      label = ndarray.swapaxes(label, 0, 1)
      batch = UtteranceBatch([self.data_name], [data],
                             [self.label_name], [label],
                             bucket_key=self.sampling[i],
                             utt_names=names,
                             utt_lens=lens)
    else:
      assert False

    self.curr_idx += 1
    return batch

def read_hdf5(filename, batching='default'):
  h5 = h5py.File(filename, "r")
  lengths = h5["seqLengths"][...].T[0].tolist()
  xin = h5['inputs'][...]
  yin = h5['targets/data']['classes'][...]
  n_out = h5['targets/size'].attrs['classes']

  utterances = []
  states = []
  offset = 0
  for length in lengths:
    utterances.append(xin[offset:offset + length])
    states.append(yin[offset:offset + length])
    offset += length
  
  names = h5['seqTags'][...]
  h5.close()
  return names, utterances, states, n_out


def get_data():
    train_n, train_x, train_y, n_out = read_hdf5('./data/train.0001')
    valid_n, valid_x, valid_y, _ = read_hdf5(args.input_file)

    sampling = args.sampling
    if sampling is not None:
      sampling = sampling.split(',')
      if len(sampling) > 1: # bucket sampling
        sampling = [ int(s) for s in sampling ]
      else:
        try:
          hills = int(sampling)
          sampling = hills
        except ValueError:
          pass

    data_train  = UtteranceIter(train_x, train_y, train_n, args.batch_size, sampling=sampling)
    data_val    = UtteranceIter(valid_x, valid_y, valid_n, args.batch_size, sampling=sampling, shuffle=False)
    return data_train, data_val, n_out

class FrameError(mx.metric.EvalMetric):
  """Calculate frame error rate."""

  def __init__(self):
    super(FrameError, self).__init__('frame-error')

  def update(self, labels, preds):
    for label, pred_label in zip(labels, preds):
      if pred_label.shape != label.shape:
        pred_label = ndarray.argmax_channel(pred_label)
      pred_label = pred_label.asnumpy().astype('int32').flatten()
      label = label.asnumpy().astype('int32').flatten()

      idx = np.where(label != -1)
      self.sum_metric += (pred_label[idx] != label[idx]).sum()
      self.num_inst += len(pred_label[idx])

class PosteriorExtraction(mx.metric.EvalMetric):
  """write frame-wise posteriors to an HDF5 output file."""

  def __init__(self, filename, names):
    super(PosteriorExtraction, self).__init__('frame-error')
    import SprintCache
    self.file = SprintCache.FileArchive(filename)
    self.cur_idx = 0
    self.names = names
    
  def finalize(self):
    self.file.finalize()

  def update(self, labels, preds):
    for label, pred_label in zip(labels, preds):
      idx = np.where(label.asnumpy() != -1)[0]
      #print idx
      pcx = pred_label.asnumpy().astype('float32')
      #print pcx[idx[0]]
      #print pcx.shape
      pred_label = np.log(pcx)[idx]
      times = zip(range(0, pred_label.shape[0]), range(1, pred_label.shape[0] + 1))
      self.file.addFeatureCache(self.names[self.cur_idx] + '/1',pred_label,times)
      self.cur_idx += 1

def train(args):
    data_train, data_val, n_out = get_data()
    if args.stack_rnn:
        cell = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell.add(mx.rnn.FusedRNNCell(args.num_hidden, num_layers=1,
                                         mode='lstm', prefix='lstm_l%d'%i,
                                         bidirectional=args.bidirectional))
            if args.dropout > 0 and i < args.num_layers - 1:
                cell.add(mx.rnn.DropoutCell(args.dropout, prefix='lstm_d%d'%i))
    else:
        cell = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, dropout=args.dropout,
                                   mode='lstm', bidirectional=args.bidirectional)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('labels')

        output, _ = cell.unroll(seq_len, inputs=data, merge_outputs=True, layout='TNC')
        pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=n_out, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax', ignore_label=-1, multi_output=True, use_ignore=True)

        return pred, ('data',), ('labels',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            cell, args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = FrameError(),
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params, 
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, args.model_prefix, 1)
                              if args.model_prefix else None)

def test(args):
    assert args.model_prefix, "Must specify path to load from"
    _, data_val, n_out = get_data()

    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                mode='lstm', bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell = mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_%dl0_'%i)
            if args.bidirectional:
                cell = mx.rnn.BidirectionalCell(
                        cell,
                        mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_%dr0_'%i),
                        output_prefix='bi_lstm_%d'%i)
            stack.add(cell)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('labels')
        stack.reset()
        output, _ = stack.unroll(seq_len, inputs=data, merge_outputs=True, layout='TNC')
        pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden * (1 + args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=n_out, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        out = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax', ignore_label=-1, multi_output=True,
                                   use_ignore=True)

        return out, ('data',), ('labels',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_val.default_key,
        context             = contexts)
    model.bind(data_val.provide_data, data_val.provide_label, for_training=False)

    # note here we load using SequentialRNNCell instead of FusedRNNCell.
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, args.model_prefix, args.load_epoch)
    model.set_params(arg_params, aux_params)
    #extr = PosteriorExtraction(args.output_file, data_val.names)
    #model.score(data_val, extr, batch_end_callback=mx.callback.Speedometer(args.batch_size, 5))
    #extr.finalize()

    import SprintCache
    import os
    if os.path.exists(args.output_file):
      os.remove(args.output_file)
    cache = SprintCache.FileArchive(args.output_file)

    for batch in data_val:
      model.forward(batch, is_train=False)
      outputs = model.get_outputs()
      pcx = np.log(outputs[0].asnumpy().astype('float32'))

      batch_size = len(batch.utt_names)
      pcx = pcx.reshape((pcx.shape[0]/batch_size,batch_size,pcx.shape[1]))
      pcx = pcx.swapaxes(0, 1)

      for i in xrange(batch_size):
        print batch.utt_names[i]
        length = batch.utt_lens[i]
        times = zip(range(0, length), range(1, length + 1))
        cache.addFeatureCache(batch.utt_names[i] + '/1', pcx[i][:length], times)

    cache.finalize()

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)
