import io
from collections import defaultdict

vocab_encoder = {}
vocab_decoder = []

def encode(object):
  if object in vocab_encoder:
    return vocab_encoder[object]
  else:
    id = len(vocab_encoder)
    vocab_encoder[object] = id
    vocab_decoder.append(object)
    assert len(vocab_encoder) == len(vocab_decoder)
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
    #out_contexts_file.write(u'{0}\t{1}\t{2}\n'.format(left_context_id, right_context_id, phrase_id))
    out_contexts_file.write(u'{0}\t{1}\t{2}\n'.format(' '.join(left_context), ' '.join(right_context), ' '.join(phrase)))

def extract_from_file(text_filename, out_contexts_filename):
  with io.open(out_contexts_filename, encoding='utf8', mode='w') as out_contexts_file:
    for line in io.open(text_filename, encoding='utf8'):
      extract_from_sentence(line.strip().split(), 1, 2, 2, out_contexts_file)
      extract_from_sentence(line.strip().split(), 2, 2, 2, out_contexts_file)
      extract_from_sentence(line.strip().split(), 3, 2, 2, out_contexts_file)

def compute_context_based_features(in_sorted_contexts_file, in_cdec_rules_file, out_cdec_rules_file, min_context_frequency, tgt_side=True):
  # map each phrase and each context into a feature value
  # initialize the maps
  phrase_id_to_val = defaultdict(float)
  context_id_to_val = defaultdict(float)
  # read the contexts file
  current_left_context_id = None
  current_right_context_id = None
  current_phrase_ids = set()
  for line in in_sorted_contexts_file:
    left_context_id, right_context_id, phrase_id = line.strip().split()
    left_context_id, right_context_id, phrase_id = int(left_context_id), int(right_context_id), int(phrase_id)
    if current_left_context_id == left_context_id and current_right_context_id == right_context_id:
      # we are still processing the same context
      current_phrase_ids.add(phrase_id)
    else:
      # before moving into a new context, we need to sum up the current one first
      if len(current_phrase_ids) >= min_context_frequency:
        # well, only if the context is frequent enough
        context_id_to_val[ (current_left_context_id, current_right_context_id) ] += 1
        for phrase_id in current_phrase_ids:
          phrase_id_to_val[phrase_id] += 1
      # now, reset your "currents"
      current_left_context_id, current_right_context_id = left_context_id, right_context_id
      current_phrase_ids.clear()
      current_phrase_ids.add(phrase_id)
  # congrats! you summarized all these contexts and phrases into features
  # now, it's time to add those features to cdec_rules
  for line in in_cdec_rules_file:
    (src_rule, tgt_rule, features) = line.strip().split('|||')
    if tgt_side:
      rule = tgt_rule
    else:
      rule = src_rule
    if rule.contains(WILDCARD):
      context_id = parse_context(rule)
      if context_id in context_id_to_val:
        features += ' IS_FREQUENT_CONTEXT={0}'.format(context_id_to_val[ context_id ])
    else:
      phrase_id = parse_phrase(rule)
      if phrase_id in phrase_id_to_val:
        features += ' IN_FREQUENT_CONTEXT={0}'.format(phrase_id_to_val[ phrase_id ])
    out_cdec_rules_file.write('{0}|||{1}|||{2}\n'.format(src_rule, tgt_rule, features))

#extract_from_file('english-ptb.tok', 'english-ptb-contexts.txt')
#exit(0)
min_context_frequency = 10
with io.open('english-ptb-contexts-sorted-filtered.txt', encoding='utf8', mode='w') as filtered:
  current_left_context_id, current_right_context_id, current_phrase_ids = None, None, set()
  phrase_id_to_val = defaultdict(float)
  context_id_to_val = defaultdict(float)
  for line in io.open('english-ptb-contexts-sorted.txt', encoding='utf8'):
    left_context_id, right_context_id, phrase_id = line.strip().split('\t')
    #left_context_id, right_context_id, phrase_id = int(left_context_id), int(right_context_id), int(phrase_id)
    if current_left_context_id == left_context_id and current_right_context_id == right_context_id:
      # we are still processing the same context
      current_phrase_ids.add(phrase_id)
    else:
      # before moving into a new context, we need to sum up the current one first
      if len(current_phrase_ids) >= min_context_frequency:
        # well, only if the context is frequent enough
        context_id_to_val[ (current_left_context_id, current_right_context_id) ] += 1
        for phrase_id in current_phrase_ids:
          filtered.write(u'{0} <0> {1} [[ {2} ]]\n'.format(current_left_context_id, current_right_context_id, phrase_id))
      # now, reset your "currents"
      current_left_context_id, current_right_context_id = left_context_id, right_context_id
      current_phrase_ids.clear()
      current_phrase_ids.add(phrase_id)
      
