from tensorflow.python.platform import gfile
import tensorflow as tf
import json
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def read_data_graph(src_path, edge_path, ref_path, wvocab, evocab, cvocab, hparams):
    data_set = []
    unks = []
    ct = 0
    lct = 0
    with tf.gfile.GFile(src_path, mode="r") as src_file:
        with tf.gfile.GFile(edge_path, mode="r") as edge_file:
            with tf.gfile.GFile(ref_path, mode="r") as ref_file:
                src, edges, ref = src_file.readline(), edge_file.readline(), ref_file.readline()

                while src and ref:
                    ct += 1

                    src_seq = src.lower().rstrip("\n").split(" ")
                    tgt = ref.lower().rstrip("\n").split(" ")
                    ref = ref.rstrip("\n")
                    graph = json.loads(edges.rstrip("\n"))

                    src_ids = []
                    tgt_ids = []
                    char_ids = []
                    unk = []
                    edges = []
                    reen = {}
                    ct_re = 0
                    i = 0
                    depth = []
                    for w in src_seq:
                        if w == " " or len(w) < 1:
                            continue
                        char_id = []
                        for cc in range(0, len(w)):
                            if w[cc] not in cvocab:
                                char_id.append(76)
                            else:
                                char_id.append(cvocab[w[cc]])
                        char_ids.append(char_id)
                        if w in wvocab:
                            src_ids.append(wvocab[w])
                            unk.append(w)
                        else:
                            src_ids.append(UNK_ID)
                            unk.append(w)
                        depth.append(0)

                        i += 1
                    depth[0] = 1

                    for w in tgt:
                        if w in wvocab:
                            tgt_ids.append(wvocab[w])
                        else:
                            tgt_ids.append(UNK_ID)

                    for l in graph:
                        id1 = int(l)
                        for pair in graph[l]:
                            edge, id2 = pair[0], pair[1]
                            if edge in evocab:
                                edge = evocab[edge]
                            else:
                                edge = UNK_ID

                            if depth[id1] == 0:
                                print("fuck")
                            if depth[int(id2)] == 0:
                                depth[int(id2)] = depth[id1] + 1
                                if depth[int(id2)] > ct_re:
                                    ct_re = depth[int(id2)]
                            edges.append([edge, id1, int(id2)])
                    data_set.append([src_ids, tgt_ids, edges, char_ids, depth, ref.rstrip("\n")])
                    unks.append(unk)

                    if not(len(src_ids) < hparams.max_src_len - 1 and len(edges) < hparams.max_src_len - 1 and len(
                            tgt_ids) < hparams.max_tgt_len - 1):
                        lct += 1
                    src, edges, ref = src_file.readline(), edge_file.readline(), ref_file.readline()
    print(lct)
    print(len(data_set))
    return data_set, unks