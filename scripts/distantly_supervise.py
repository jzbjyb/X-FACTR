import json
import time
from collections import defaultdict
import os
import random

import sling
import sling.task.workflow as workflow

random.seed(111)

class SlingExtractor(object):

  """ Extract passages from Wikipedia upto a maximum length which mention a
  fact from Wikidata.
  """

  def load_kb(self,
              kb_file="local/data/e/wiki/kb.sling"):
    """ Load self.kb """
    print("loading and indexing kb...")
    start = time.time()
    self.kb = sling.Store()
    self.kb.load(kb_file)
    self.kb.freeze()
    self.extract_property_names()
    print("loading took",(time.time() - start),"sec")

  def extract_property_names(self):
    """ Store names of properties in a dict """
    # TODO: add property aliases here
    print("storing property names...")
    start = time.time()
    self.property_names = defaultdict(list)
    for frame in self.kb:
      if "id" in frame and frame.id.startswith("P"):
        self.property_names[frame.id].append(frame.name)
    print("found", str(len(self.property_names)), "properties")
    print("took",(time.time() - start),"sec")

  def load_corpus(self,
                  corpus_file="local/data/e/wiki/en/documents-00000-of-00010.rec"):
    """ Load self.corpus """
    print("loading the corpus...")
    self.corpus = sling.Corpus(corpus_file)
    self.commons = sling.Store()
    self.docschema = sling.DocumentSchema(self.commons)
    self.commons.freeze()

  def annotate_corpus(
      self,
      unannotated_file="local/data/e/wiki/en/documents-00000-of-00010.rec",
      annotated_file="/tmp/labeled.rec"):
    if os.path.exists(annotated_file):
      return
    labeler = entity.EntityWorkflow("wiki-label")
    unannotated = labeler.wf.resource(unannotated_file, format="records/document")
    annotated = labeler.wf.resource(annotated_file, format="records/document")
    labeler.label_documents(indocs=unannotated, outdocs=annotated)
    workflow.run(labeler.wf)

  def get_linked_entity(self, mention):
    """ Returns the string ID of the linked entity for this mention """
    if "evokes" not in mention.frame:
      return None
    if type(mention.frame["evokes"]) != sling.Frame:
      return None
    if "is" in mention.frame["evokes"]:
      if type(mention.frame["evokes"]["is"]) != sling.Frame:
        if ("isa" in mention.frame["evokes"] and
            mention.frame["evokes"]["isa"].id == "/w/time" and
            type(mention.frame["evokes"]["is"]) == int):
          return mention.frame["evokes"]["is"]
        else:
          return None
      else:
        return mention.frame["evokes"]["is"].id
    return mention.frame["evokes"].id

  def get_frame_id(self, frame):
    """ Returns the WikiData identifier for a property or entity """
    if "id" in frame:
      return frame.id
    if "is" in frame:
      if type(frame["is"]) != sling.Frame:
        return None
      if "id" in frame["is"]:
        return frame["is"].id
    return None

  def get_date_property(self, prop, tail):
    """ Returns date if property accepts '/w/time' as target """
    if "target" not in prop:
      return None
    if prop.target.id != "/w/time":
      return None
    prop_id = self.get_frame_id(prop)
    if type(tail) == int:
      return (prop_id, tail)
    elif type(tail) == sling.Frame and "is" in tail and type(tail["is"]) == int:
      return (prop_id, tail["is"])
    return None

  def get_canonical_property(self, prop, tail):
    """ Returns true if the property and tail are canonical WikiData properties """
    # TODO: Identify quantity mentions in text and catch corresponding facts
    # here
    if type(prop) != sling.Frame or type(tail) != sling.Frame:
      return None
    prop_id = self.get_frame_id(prop)
    tail_id = self.get_frame_id(tail)
    if prop_id is None:
      return None
    if tail_id is None:
      return None
    if not prop_id.startswith("P") or not tail_id.startswith("Q"):
      return None
    return (prop_id, tail_id)

  def print_relation(self, relation):
    """ Print the distantly supervised relation mention """
    print(relation[2],"::", ",".join(relation[3]), "::", relation[1], "::", relation[0])

  def serialize_relation(self, document, tok_to_char_offset, ctx_span,
                         subj_id, subj_mentions, property_id,
                         tail_id, tail_mention, ctx_type):
    """ Create JSON object with distantly supervised fact """
    context = " ".join(
        [tt.word for tt in document.tokens[ctx_span[0]:ctx_span[1]]])
    my_offsets = {}
    begin = tok_to_char_offset[ctx_span[0]]
    for ix in range(ctx_span[0], ctx_span[1]):
      my_offsets[ix] = tok_to_char_offset[ix] - begin
    subject = {"wikidata_id": subj_id,
               "mentions": [{"start": my_offsets[m.begin],
                             "text": " ".join(
                                 tt.word for tt in document.tokens[m.begin:m.end])}
                            for m in subj_mentions],
               }
    if property_id not in self.property_names:
      print("did not find", property_id, "in names")
    relation = {"wikidata_id": property_id,
                "text": self.property_names.get(property_id, property_id)}
    if tail_id is None:
      obj = None
      id_ = subj_id + "_" + property_id + "_None"
    else:
      obj = {"wikidata_id": tail_id,
             "mention": {
                 "start": my_offsets[tail_mention.begin],
                 "text": " ".join(
                     tt.word for tt in document.tokens[
                         tail_mention.begin:tail_mention.end])}
             }
      id_ = subj_id + "_" + property_id + "_" + str(tail_id)
    serialized = json.dumps({
        "id": id_,
        "context": context,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "context_type": ctx_type,
    })
    return serialized

  def init_stats(self):
    """ Initialize dicts to hold relation counts """
    self.relation_stats = {"sentences": defaultdict(int),
                           "paragraphs": defaultdict(int),
                           "documents": defaultdict(int),
                           "entity negatives": defaultdict(int)}

  def link_documents(self, N=None, out_file="/tmp/linked.rec",
                     add_negatives=False, filter_subjects=None):
    """ Load n documents and link them to facts """
    start = time.time()
    fout = open(out_file, "w")
    for n, (doc_id, doc_raw) in enumerate(self.corpus.input):
      if n == N:
        break
      if n % 1000 == 0:
        print("processed", n, "items in %.1f" % (time.time() - start), "seconds")
      # get kb items
      doc_id = str(doc_id, "utf-8")
      if filter_subjects is not None and doc_id not in filter_subjects:
        continue
      kb_item = self.kb[doc_id]
      tail_entities = {}
      all_properties = []
      for prop, tail in kb_item:
        tup = self.get_canonical_property(prop, tail)
        if tup is None:
          tup = self.get_date_property(prop, tail)
          if tup is None:
            continue
        tail_entities[tup[1]] = tup[0]
        all_properties.append(tup[0])
      store = sling.Store(self.commons)
      document = sling.Document(store.parse(doc_raw), store, self.docschema)
      if len(document.tokens) == 0:
        print("Skipping %s No tokens." % (doc_id))
        continue
      # build token maps
      tok_to_sent_id, tok_to_para_id, sent_to_span, para_to_span = {}, {}, {}, {}
      tok_to_char_offset = {}
      offset = 0
      sent_begin = para_begin = 0
      sent_id = para_id = 0
      for ii, token in enumerate(document.tokens):
        if ii > 0 and token.brk == 4:
          para_to_span[para_id] = (para_begin, ii)
          sent_to_span[sent_id] = (sent_begin, ii)
          para_id += 1
          sent_id += 1
          sent_begin = para_begin = ii
        elif ii > 0 and token.brk == 3:
          sent_to_span[sent_id] = (sent_begin, ii)
          sent_id += 1
          sent_begin = ii
        tok_to_sent_id[ii] = sent_id
        tok_to_para_id[ii] = para_id
        tok_to_char_offset[ii] = offset
        offset += len(token.word) + 1
      para_to_span[para_id] = (para_begin, ii+1)
      sent_to_span[sent_id] = (sent_begin, ii+1)
      # find subjects
      sent_to_subj, para_to_subj = defaultdict(list), defaultdict(list)
      mentid_to_linked_entity = {}
      sorted_mentions = sorted(document.mentions, key=lambda m: m.begin)
      for ii, mention in enumerate(sorted_mentions):
        if tok_to_sent_id[mention.begin] != tok_to_sent_id[mention.end-1]:
          continue
        linked_entity = self.get_linked_entity(mention)
        mentid_to_linked_entity[ii] = linked_entity
        if linked_entity == doc_id:
          sent_id = tok_to_sent_id[mention.begin]
          sent_to_subj[sent_id].append(mention)
          para_id = tok_to_para_id[mention.begin]
          para_to_subj[para_id].append(mention)

      # find tails
      relations = []
      seen_properties = {}
      for ii, mention in enumerate(sorted_mentions):
        # first look for sentence matches
        linked_entity = mentid_to_linked_entity[ii]
        if linked_entity == doc_id:
          continue
        if linked_entity in tail_entities:
          if tail_entities[linked_entity] in seen_properties:
            continue
          my_sent = tok_to_sent_id[mention.begin]
          if my_sent in sent_to_subj:
            my_para = tok_to_para_id[mention.begin]
            para_span = para_to_span[my_para]
            #sent_span = sent_to_span[my_sent]
            fout.write(self.serialize_relation(
                document, tok_to_char_offset, para_span, doc_id,
                sent_to_subj[my_sent], tail_entities[linked_entity],
                linked_entity, mention, "sentence") + "\n")
            seen_properties[tail_entities[linked_entity]] = my_para
            self.relation_stats["sentences"][tail_entities[linked_entity]] += 1

      for ii, mention in enumerate(sorted_mentions):
        # next look for paragraph matches
        linked_entity = mentid_to_linked_entity[ii]
        if linked_entity == doc_id:
          continue
        if linked_entity in tail_entities:
          if tail_entities[linked_entity] in seen_properties:
            continue
          my_para = tok_to_para_id[mention.begin]
          if my_para in para_to_subj:
            para_span = para_to_span[my_para]
            fout.write(self.serialize_relation(
                document, tok_to_char_offset, para_span, doc_id,
                para_to_subj[my_para], tail_entities[linked_entity],
                linked_entity, mention, "paragraph") + "\n")
            seen_properties[tail_entities[linked_entity]] = my_para
            self.relation_stats["paragraphs"][tail_entities[linked_entity]] += 1

      # add negatives
      if add_negatives:
        max_neg = len(seen_properties)
        num_neg = 0
        all_para_id = list(para_to_subj.keys())
        if not all_para_id:
          continue
        for tail, prop in tail_entities.items():
          if num_neg == max_neg:
            break
          if prop in seen_properties:
            continue
          random_para_id = random.choice(all_para_id)
          random_para_span = para_to_span[random_para_id]
          fout.write(self.serialize_relation(
              document, tok_to_char_offset, random_para_span, doc_id,
              para_to_subj[random_para_id], prop,
              None, None, "entity negative") + "\n")
          num_neg += 1
          seen_properties[prop] = None
          self.relation_stats["entity negatives"][prop] += 1

    fout.close()
    print("Sentences -- Total ", sum(self.relation_stats["sentences"].values()))
    print(" :: ".join(
        "%s:%d" % (k, v) for k, v in self.relation_stats["sentences"].items()))
    print("Paragraphs -- Total ", sum(self.relation_stats["paragraphs"].values()))
    print(" :: ".join(
        "%s:%d" % (k, v) for k, v in self.relation_stats["paragraphs"].items()))
