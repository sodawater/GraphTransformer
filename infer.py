from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import shutil
import hashlib
from sys import platform
import data_utils
from data_utils import *
import argparse
import copy
import collections
from gensim.models import KeyedVectors
from graph_transformer import GraphTransformer

import json
FLAGS = None
# tf.enable_eager_execution()
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="model/graph/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="2", help="which gpu to use")

    parser.add_argument("--train_data", type=str, default="training",
                        help="Training data path")

    parser.add_argument("--valid_data", type=str, default="dev",
                        help="Valid data path")

    parser.add_argument("--test_data", type=str, default="test",
                        help="Test data path")

    parser.add_argument("--from_vocab", type=str, default="data/vocab_15000",
                        help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="data/vocab_15000",
                        help="to vocab path")
    parser.add_argument("--label_vocab", type=str, default="data/evocab",
                        help="label vocab path")
    parser.add_argument("--output_dir", type=str, default="model/graph/")


    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")

    parser.add_argument("--from_vocab_size", type=int, default=15000, help="source vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=15000, help="target vocabulary size")
    parser.add_argument("--edge_vocab_size", type=int, default=150, help="edge label vocabulary size")
    parser.add_argument("--enc_layers", type=int, default=10, help="Number of layers in the encoder")
    parser.add_argument("--dec_layers", type=int, default=4, help="Number of layers in the decoder")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="Size of each model layer")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--max_src_len", type=int, default=100, help="Maximum length of source ordering")
    parser.add_argument("--max_tgt_len", type=int, default=100, help="Maximum length of target ordering")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--use_copy", type=int, default=True, help="Whether use copy mechanism")
    parser.add_argument("--use_depth", type=int, default=False, help="Whether use depth embedding")
    parser.add_argument("--use_charlstm", type=int, default=False, help="Whether use character embedding")
    parser.add_argument("--input_keep_prob", type=float, default=1.0, help="Dropout input keep prob")
    parser.add_argument("--output_keep_prob", type=float, default=0.9, help="Dropout output keep prob")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.5)

def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        edge_vocab_size=flags.edge_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,
        emb_dim=flags.emb_dim,
        max_train_data_size=flags.max_train_data_size,

        train_data=flags.train_data,
        valid_data=flags.valid_data,
        test_data=flags.test_data,

        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,
        label_vocab=flags.label_vocab,
        share_vocab=False,

        # model params
        use_copy=flags.use_copy,
        use_depth=flags.use_depth,
        use_charlstm=flags.use_charlstm,
        input_keep_prob=flags.input_keep_prob,
        output_keep_prob=flags.output_keep_prob,
        dropout_rate=flags.dropout_rate,
        init_weight=0.1,
        num_units=flags.num_units,
        num_heads=flags.num_heads,
        enc_layers=flags.enc_layers,
        dec_layers=flags.dec_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        max_src_len=flags.max_src_len,
        max_tgt_len=flags.max_tgt_len,
        max_seq_length=100,
        #train params
        epoch_num=flags.epoch_num,
        epoch_step=0,
        lambda1=flags.lambda1,
        lambda2=flags.lambda2
    )

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  # GPU options:
  # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto


class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
  pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
  pass

def create_model(hparams, model, length=22):
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)

    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)

def read_data(src_path, tgt_path, vocab):

    data_set = []
    with tf.gfile.GFile(src_path, mode="r") as src_file:
        with tf.gfile.GFile(tgt_path, mode="r") as tgt_file:
            src, tgt = src_file.readline(), tgt_file.readline()
            while src and tgt:
                src_ids = [int(x) for x in src.rstrip("\n").split(" ")]
                tgt_ids = [int(x) for x in tgt.rstrip("\n").split(" ")]

                pair = (src_ids, tgt_ids)
                data_set.append(pair)
                src, tgt = src_file.readline(), tgt_file.readline()
    return data_set

def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def infer(hparams):

    embeddings = init_embedding(hparams)
    hparams.add_hparam(name="embeddings", value=embeddings)
    wvocab, wvocab_rev = initialize_vocabulary(hparams.from_vocab)
    evocab, evocab_rev = initialize_vocabulary(hparams.label_vocab)
    cvocab, cvocab_rev = initialize_vocabulary("data/cvocab")

    test_data, test_unks = read_data_graph("data/test.src", "data/test.edge",
                                             "data/test.tgt",
                                           wvocab, evocab, cvocab, hparams)

    to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(hparams.from_vocab)
    outputs = []

    train_model, eval_model, infer_model = create_model(hparams, GraphTransformer)
    config = get_config_proto(
        log_device_placement=False)
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)


    ckpt = tf.train.get_checkpoint_state(hparams.train_dir)
    ckpt_path = os.path.join(hparams.train_dir, "ckpt")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        global_step = train_model.model.global_step.eval(session=train_sess)
        infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
        
        f1 = open("ref_file" + str(global_step), "w", encoding="utf-8")
        f2 = open("predict_file" + str(global_step), "w", encoding="utf-8")
        ct = 0
        for id in range(0, int(len(test_data) / hparams.batch_size) + 1):
            given, predict, align = infer_model.model.infer_step_beam(infer_sess, test_data.copy(), no_random=True,
                                                                      id=id * hparams.batch_size)
            for i in range(hparams.batch_size):
                sample_output = predict[i][1:]

                if hparams.EOS_ID in sample_output:
                    sample_output = sample_output[:sample_output.index(hparams.EOS_ID)]


                _, _, _, _, _, answer = test_data[(id * hparams.batch_size + i) % len(test_data)]
                unks = test_unks[(id * hparams.batch_size + i) % len(test_data)]
                f1.write(answer + "\n")
                outputs = []
                unk = 0
                for output in sample_output:
                    w = tf.compat.as_str(rev_to_vocab[output])
                    if w == "_UNK":
                        y = unks[align[i][unk]]
                        outputs.append(y)
                    elif w != "_UNK":
                        outputs.append(w)
                    unk += 1

                s = " ".join(outputs)
                f2.write(s + "\n")

                ct += 1
                if ct >= len(test_data):
                    break

        f1.close()
        f2.close()


    else:
        raise ValueError("ckpt file not found.")




def init_embedding(hparams):
    f = open(hparams.from_vocab, "r", encoding="utf-8")
    vocab = []
    for line in f:
        vocab.append(line.rstrip("\n"))

    word_vectors = KeyedVectors.load_word2vec_format("data/amr_vector.txt")

    emb = []
    num = 0
    for i in range(0, len(vocab)):
        word = vocab[i]
        if word in word_vectors:
            num += 1
            emb.append(word_vectors[word])
        else:
            emb.append((0.1 * np.random.random([hparams.emb_dim]) - 0.05).astype(np.float32))
    print(" init embedding finished")
    emb = np.array(emb)
    # print(num)
    # print(emb.shape)
    return emb

def main(_):

    hparams = create_hparams(FLAGS)
    infer(hparams)

if __name__ == "__main__":
    print(FLAGS)
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.train_dir = FLAGS.model_dir + FLAGS.train_dir
    FLAGS.output_dir = FLAGS.out_dir + FLAGS.output_dir
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()
