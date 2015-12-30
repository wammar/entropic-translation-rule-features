import gzip
import re
import time
import io
import math
import os
import sys
import argparse
from collections import defaultdict
from subprocess import call

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--src_monolingual_filename")
argParser.add_argument("-t", "--tgt_monolingual_filename")
argParser.add_argument("-i", "--input_cdec_rules_dirname")
argParser.add_argument("-o", "--output_cdec_rules_dirname")
argParser.add_argument("-np", "--min_phrase_length", type=int, default=1)
argParser.add_argument("-xp", "--max_phrase_length", type=int, default=1)
argParser.add_argument("-ml", "--min_left_context_length", type=int, default=1)
argParser.add_argument("-xl", "--max_left_context_length", type=int, default=1)
argParser.add_argument("-mr", "--min_right_context_length", type=int, default=1)
argParser.add_argument("-xr", "--max_right_context_length", type=int, default=1)
argParser.add_argument("-mf", "--min_context_frequency", type=int, default=3)
args = argParser.parse_args()

vocab_encoder = {}
vocab_decoder = []

def encode(object):
  if object == u'about':
    print '"object" is being encoded as ...' 
  if object in vocab_encoder:
    return vocab_encoder[object]
  else:
    id = len(vocab_encoder)
    vocab_encoder[object] = id
    vocab_decoder.append(object)
    assert len(vocab_encoder) == len(vocab_decoder)
    if object == u'about': print vocab_encoder[u'about']
    return id

def decode(id):
  return vocab_decoder[id]

def extract_from_sentence(tokens, phrase_size, left_context_size, right_context_size, out_contexts_file):
  # for each position in the sentence
  for phrase_start_pos in xrange(len(tokens) - phrase_size + 1):
    # identify the left context
    left_context = tokens[phrase_start_pos - left_context_size:phrase_start_pos]
    if len(left_context) < left_context_size:
      left_context.insert(0, '<sentence-boundary>')
    left_context_id = encode(tuple(left_context))
    # identify the right context
    right_context = tokens[phrase_start_pos + phrase_size:phrase_start_pos + phrase_size + right_context_size]
    if len(right_context) < right_context_size:
      right_context.append('<sentence-boundary>')
    right_context_id = encode(tuple(right_context))
    # identify the phrase
    phrase = tokens[phrase_start_pos:phrase_start_pos + phrase_size]
    phrase_id = encode(tuple(phrase))
    # write this context to file
    out_contexts_file.write(u'{0}\t{1}\t{2}\n'.format(left_context_id, right_context_id, phrase_id))
    #out_contexts_file.write(u'{0}\t{1}\t{2}\n'.format(' '.join(left_context), ' '.join(right_context), ' '.join(phrase)))

def extract_from_file(text_filename, out_contexts_filename):
  with io.open(out_contexts_filename, encoding='utf8', mode='w') as out_contexts_file:
    for line in io.open(text_filename, encoding='utf8'):
      for phrase_length in range(args.min_phrase_length, args.max_phrase_length + 1):
        for left_context_length in range(args.min_left_context_length, args.max_left_context_length + 1):
          for right_context_length in range(args.min_right_context_length, args.max_right_context_length + 1):
            extract_from_sentence(line.strip().split(), phrase_length, left_context_length, right_context_length, out_contexts_file)

def read_contexts_file(in_sorted_contexts_file, min_context_frequency):
  # the job of this method is to summarize the observed contexts into the following two maps and return them
  phrase_id_to_val, context_id_to_val = defaultdict(float), defaultdict(float)
  # read the contexts file
  current_left_context_id = None
  current_right_context_id = None
  current_phrase_ids = set()
  max_context_val, max_phrase_val = 1.0, 1.0
  for line in in_sorted_contexts_file:
    left_context_id, right_context_id, phrase_id = line.strip().split('\t')
    left_context_id, right_context_id, phrase_id = int(left_context_id), int(right_context_id), int(phrase_id)
    if current_left_context_id == left_context_id and current_right_context_id == right_context_id:
      # we are still processing the same context
      current_phrase_ids.add(phrase_id)
    else:
      # before moving into a new context, we need to sum up the current one first
      if len(current_phrase_ids) >= min_context_frequency:
        # well, only if the context is frequent enough
        # a context constitutes a good rule if it appears 
        current_context_val = math.log(len(current_phrase_ids))
        context_id_to_val[ (current_left_context_id, current_right_context_id) ] = current_context_val
        max_context_val = max(max_context_val, current_context_val)
        for phrase_id in current_phrase_ids:
          current_phrase_val = phrase_id_to_val[phrase_id] + 1
          phrase_id_to_val[phrase_id] = current_phrase_val
          max_phrase_val = max(max_phrase_val, current_phrase_val)
      # now, reset your "currents"
      current_left_context_id, current_right_context_id = left_context_id, right_context_id
      current_phrase_ids.clear()
      current_phrase_ids.add(phrase_id)
  # normalize context_id features
  for context_id in context_id_to_val.keys():
    context_id_to_val[context_id] *= 1.0 / max_context_val
  # take logs of phrase_id features as well, and normalize
  max_phrase_val = math.log(max_phrase_val)
  for phrase_id in phrase_id_to_val.keys():
    phrase_id_to_val[phrase_id] = math.log( phrase_id_to_val[phrase_id] ) / max_context_val
  return (phrase_id_to_val, context_id_to_val)

lexical_regex = re.compile(r'[^\[]+')
surrounded_phrase_regex = re.compile(r'(\[X,\d+\]) ([^\[]+) (\[X,\d+\])$')
surrounding_context_regex = re.compile(r'([^\[]+) (\[X,\d+\]) ([^\[]+)$')
def parse_rule_side(rule_side):
  # is it a simple phrase?
  if lexical_regex.match(rule_side) and tuple(rule_side.split()) in vocab_encoder:
    return (vocab_encoder[tuple(rule_side.split())], None)
  # a phrase surrounded with context?
  match = surrounded_phrase_regex.match(rule_side)
  if match and tuple(match.groups()[1].split()) in vocab_encoder:
    return (vocab_encoder[tuple(match.groups()[1].split())], None)
  # a context surrounding a gap?
  match = surrounding_context_regex.match(rule_side)
  if match and \
        tuple(match.groups()[0].split()) in vocab_encoder and \
        tuple(match.groups()[2].split()) in vocab_encoder:
    return (None, (vocab_encoder[tuple(match.groups()[0].split())], vocab_encoder[tuple(match.groups()[2].split())]) )
  
  # never mind!
  return (None, None)

def compute_context_based_features(in_src_sorted_contexts_file, 
                                   in_tgt_sorted_contexts_file, 
                                   in_cdec_rules_file, 
                                   out_cdec_rules_file, 
                                   min_context_frequency):
  # map each phrase and each context into a feature value
  # initialize the maps from monolingual phrases/contexts to feature values
  tgt_phrase_id_to_val, tgt_context_id_to_val = read_contexts_file(in_tgt_sorted_contexts_file, min_context_frequency)
  src_phrase_id_to_val, src_context_id_to_val = read_contexts_file(in_src_sorted_contexts_file, min_context_frequency)
  assert None not in src_phrase_id_to_val and \
      None not in src_context_id_to_val and \
      None not in tgt_phrase_id_to_val and \
      None not in tgt_context_id_to_val

  # congrats! you summarized all these contexts and phrases into monolingual features
  # now, it's time to add those features to cdec_rules
  for line in in_cdec_rules_file:
    # read the src/tgt  context-/phrase- id in each rule
    if len(line.strip()) == 0:
      continue
    x, src_rule, tgt_rule, features, alignment = line.strip().split(' ||| ')
    src_phrase_id, src_context_id = parse_rule_side(src_rule)
    tgt_phrase_id, tgt_context_id = parse_rule_side(tgt_rule)
    # what features do we have for this rule?
    src_phrase_feat, src_context_feat, tgt_phrase_feat, tgt_context_feat, bilingual_phrase_feat, bilingual_context_feat = 0, 0, 0, 0, 0, 0
    if src_phrase_id in src_phrase_id_to_val:
      src_phrase_feat = src_phrase_id_to_val[src_phrase_id]
      if src_phrase_feat != 0:      
        features += ' SRC_PHRASE=' + str(src_phrase_feat)
    if src_context_id in src_context_id_to_val:
      src_context_feat = src_context_id_to_val[src_context_id]
      features += ' SRC_CONTEXT=' + str(src_context_feat)
    if tgt_phrase_id in tgt_phrase_id_to_val:
      tgt_phrase_feat = tgt_phrase_id_to_val[tgt_phrase_id]
      if tgt_phrase_feat != 0:
        features += ' TGT_PHRASE=' + str(tgt_phrase_feat)
    if tgt_context_id in tgt_context_id_to_val:
      tgt_context_feat = tgt_context_id_to_val[tgt_context_id]
      features += ' TGT_CONTEXT=' + str(tgt_context_feat)
    if tgt_phrase_feat != 0 and src_phrase_feat != 0:
      bilingual_phrase_feat = tgt_phrase_feat * src_phrase_feat
      features += ' BI_PHRASE=' + str(bilingual_phrase_feat)
    if tgt_context_feat != 0 and src_context_feat != 0:
      bilingual_context_feat = tgt_context_feat * src_context_feat
      features += ' BI_CONTEXT=' + str(bilingual_context_feat)
    # write the rule to output file
    out_cdec_rules_file.write(u'{0} ||| {1} ||| {2} ||| {3} ||| {4}\n'.format(x, src_rule, tgt_rule, features, alignment))

# extract contexts
src_contexts_filename =  'src.contexts'
extract_from_file(args.src_monolingual_filename, src_contexts_filename)
tgt_contexts_filename = 'tgt.contexts'
extract_from_file(args.tgt_monolingual_filename, tgt_contexts_filename)

# sort and uniq
sorted_src_contexts_filename = src_contexts_filename + '.sorted'
os.system("cat " + src_contexts_filename + " | sort -k1,1 -k2,2 -k3,3 | uniq > " + sorted_src_contexts_filename)
sorted_tgt_contexts_filename = tgt_contexts_filename + '.sorted'
os.system("cat " + tgt_contexts_filename + " | sort -k1,1 -k2,2 -k3,3 | uniq > " + sorted_tgt_contexts_filename)

# now, compute a measure of contextual ambiguity for individual contexts and for individual phrases in both src and tgt sides
args.output_cdec_rules_dirname = os.path.abspath(args.output_cdec_rules_dirname)
args.input_cdec_rules_dirname = os.path.abspath(args.input_cdec_rules_dirname)
if not os.path.exists(args.output_cdec_rules_dirname):
  os.makedirs(args.output_cdec_rules_dirname)
for filename in os.listdir(args.input_cdec_rules_dirname):
  input_filename = os.path.join(args.input_cdec_rules_dirname, filename)
  output_filename = os.path.join(args.output_cdec_rules_dirname, filename)
  input_file = gzip.open(input_filename)
  input_file_content = input_file.read()
  input_file.close()
  input_file_lines = input_file_content.decode('utf8').split('\n')
  compute_context_based_features(io.open(sorted_src_contexts_filename, encoding='utf8'), 
                                 io.open(sorted_tgt_contexts_filename, encoding='utf8'),
                                 input_file_lines,
                                 io.open(output_filename, encoding='utf8', mode='w'), 
                                 args.min_context_frequency)
  print 'produced ', output_filename
  
