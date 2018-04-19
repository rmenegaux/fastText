/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <stdexcept>
#include <queue>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const char Dictionary::BOS = '>';
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args) : args_(args),
  word2int_(MAX_VOCAB_SIZE, -1), size_(0), nwords_(0), nlabels_(0),
  nsequences_(0),
  ntokens_(0), pruneidx_size_(-1) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in) : args_(args),
  size_(0), nwords_(0), nsequences_(0), nlabels_(0), ntokens_(0), pruneidx_size_(-1) {
  load(in);
}

int32_t Dictionary::find(const std::string& w) const {
  return find(w, hash(w));
}

int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  // while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
  //   id = (id + 1) % word2intsize;
  // }
  return id;
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  // if (word2int_[h] == -1) {
  //   entry e;
  //   e.word = w;
  //   e.count = 1;
  //   e.type = getType(w);
  //   words_.push_back(e);
  //   word2int_[h] = size_++;
  // } else {
  //   words_[word2int_[h]].count++;
  // }
}

// Add sequence to the dictionary
void Dictionary::add(entry e) {
  nsequences_++;
  // Find label
  e.label = findLabel(e.name);
  sequences_.push_back(e);
}

std::string Dictionary::findLabel(const std::string& name) {
  std::string label;
  auto it = name2label_.find(name);
  if (it != name2label_.end()) {
    addLabel(it->second);
    return it->second;
  }
  // Add label to dict
  // Maybe argument with default label?
  name2label_[name] = name;
  addLabel(name);
  return name;
}

// Returns label index of position
int Dictionary::labelFromPos(const std::streampos& pos) {
  int i = 0;
  // Check if position is greater than file size
  while (i < nsequences_-1 && pos > sequences_[i+1].name_pos) {
    i++;
  }
  if (pos < sequences_[i].seq_pos) {
    return -1; // Position is in the sequence name
  }
  // std::cerr << "\rPos: " << pos << std::endl;
  // std::cerr << "\rSeq: " << i << std::endl;
  // std::cerr << "\rLabel: " << sequences_[i].label << std::endl;
  // std::cerr << "\rIndex: " << label2int_[sequences_[i].label] << std::endl;
  return label2int_[sequences_[i].label];
}

void Dictionary::addLabel(const std::string& label) {
  auto it = label2int_.find(label);
  if (it == label2int_.end()) {
    label2int_[label] = nlabels_++;
  }
}

int32_t Dictionary::nwords() const {
  // FIXME
  return 1 << 2*args_->minn;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(const std::string& word,
                           std::vector<int32_t>& ngrams,
                           std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    // substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams, substrings);
  }
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeSubwords(const std::string& word,
                               std::vector<int32_t>& ngrams,
                               std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
        substrings.push_back(ngram);
      }
    }
  }
}

void Dictionary::computeSubwords(const std::string& word,
                                 std::vector<int32_t>& ngrams) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
      }
    }
  }
}

int8_t Dictionary::base2int(const char c) const {
  // With this convention, the complementary basepair is
  // (base + 2) % 4
  switch(c) {
    case 'A' : return 0;
    case 'C' : return 1;
    case 'T' : return 2;
    case 'G' : return 3;
  }
  throw std::invalid_argument("Non-ACGT character in base2int");
}

char Dictionary::int2base(const int c) const {
  switch(c) {
    case 0 : return 'A';
    case 1 : return 'C';
    case 2 : return 'T';
    case 3 : return 'G';
  }
  throw std::invalid_argument("Number greater than 3 in int2base");
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const
{
  int c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

bool Dictionary::readSequence(std::istream& in,
                              std::vector<int32_t>& ngrams,
                              std::vector<int32_t>& ngrams_comp,
                              const int length) const {
  std::queue<int8_t> queue;
  int c;
  int8_t val, val_comp, prev_val;
  int32_t index = 0, index_comp = 0;
  int32_t mult = 1;
  const int k = args_->minn;
  ngrams.clear();
  ngrams_comp.clear();

  std::streambuf& sb = *in.rdbuf();

  int i = 0;
  if (length < k) {
    return 0;
  }
  while (i < length) {
    if (i >= k) {
      ngrams.push_back(index);
      ngrams_comp.push_back(index_comp);
    }
    c = sb.sbumpc();
    if (c == '>' || c == EOF || c == ' ') {
      // Reached end of sequence
      return (i >= k);
    }
    c = toupper(c);
    if (c == 'A' || c == 'C' || c == 'G' || c == 'T') {
      val = base2int(c);
      val_comp = (val + 2) % 4;
      queue.push(val);
      if (i < k) {
        index = index * 4 + val;
        index_comp = val_comp * mult + index_comp;
        if (i < k-1) mult = mult << 2;
      }
      else {
        prev_val = queue.front();
        queue.pop();
        index = (index - prev_val * mult) * 4 + val;
        index_comp = index_comp / 4 + val_comp * mult;
      }
      i++;
    }
  }
  if (i >= k) {
    ngrams.push_back(index);
    ngrams_comp.push_back(index_comp);
    return true;
  }
  return false;
}

int32_t Dictionary::readSequence(std::string& word,
                            std::vector<int32_t>& ngrams) const {
  return 0;
}

std::string Dictionary::getSequence(int32_t index) const {
  std::string seq;
  for(int i = 0; i < args_->minn; i++) {
    // FIXME use push_back with other arithmetic?
    seq.insert(seq.begin(), int2base(index % 4));
    index = index / 4;
  }
  return seq;
}

void Dictionary::readFromFasta(std::istream& in) {
  std::string line, name;
  entry e;
  e.count = 0;
  std::streampos prev_pos = 0;
  while(std::getline(in, line).good()){
    if(line.empty() || line[0] == BOS ){ // Identifier marker
      if( !e.name.empty() ){
        add(e);
        if (args_->verbose > 1) {
          std::cerr << "\rRead sequence n" << nsequences_ << ", " << e.name << "      " <<std::flush;
        }
        e.name.clear();
        e.count = 0;
      }
      if( !line.empty() ){
        e.name = line.substr(1);
        e.seq_pos = in.tellg();
        e.name_pos = prev_pos;
      }
    } else {
      e.count += line.length();
    }
    prev_pos = in.tellg();
  }

  if( !e.name.empty() ){ // Add the last entry
    add(e);
  }

  if (args_->verbose > 0) {
    std::cerr << "\rRead sequence n" << nsequences_ << ", " << e.name << "       " << std::endl;
    std::cerr << "\rNumber of sequences " << nsequences_ << std::endl;
    std::cerr << "\rNumber of labels: " << nlabels() << std::endl;
    std::cerr << "\rNumber of words: " << nwords() << std::endl;
    // FIXME print total length
    // printDictionary();
  }
  // std::vector<int32_t> ngrams, ngrams_comp;
  // in.clear();
  // in.seekg(sequences_[0].seq_pos);
  // readSequence(in, ngrams, ngrams_comp, 20);
  // in.clear();
  // in.seekg(std::streampos(0));
  // std::cerr << "\rTEST: Ground truth" << std::endl;
  // std::getline(in, line);
  // std::getline(in, line);
  // std::cerr << line.substr(0, 20) << std::endl;
  // std::cerr << "\rSequences " << std::endl;
  // std::cerr << getSequence(ngrams[0]) << std::endl;
  // std::cerr << getSequence(ngrams[10]) << std::endl;
  // std::cerr << "\rReverse sequences " << std::endl;
  // std::cerr << getSequence(ngrams_comp[0]) << std::endl;
  // std::cerr << getSequence(ngrams_comp[10]) << std::endl;
  // if (size_ == 0) {
  //   throw std::invalid_argument(
  //       "Empty vocabulary. Try a smaller -minCount value.");
  // }
}

// FUTURE TESTS
// FindLabel
// std::streampos pos(0);
// std::cerr << "\rPosition " << pos << " has label " << findLabel(pos) << std::endl;
// pos = 1037010900;
// std::cerr << "\rPosition " << pos << " has label " << findLabel(pos) << std::endl;
// pos = 1087195199;
// std::cerr << "\rPosition " << pos << " has label " << findLabel(pos) << std::endl;
// pos = 1087195197;
// std::cerr << "\rPosition " << pos << " has label " << findLabel(pos) << std::endl;
// nlabels = length label2int_
// Assert kmers from readSequence are good

void Dictionary::printDictionary() const {
  if (args_->verbose > 1) {
  for (auto it = sequences_.begin(); it != sequences_.end(); ++it) {
    std::cerr << it->name << " " << it->name_pos << " " << it->seq_pos << " length " << it->count << " label " << it->label << " name " << it->name << std::endl;
  }
  // for (auto it = name2label_.begin(); it != name2label_.end(); ++it) {
  //   std::cerr << it->first << " " << it->second << std::endl;
  // }
  // for (auto it = label2int_.begin(); it != label2int_.end(); ++it) {
  //   std::cerr << it->first << " " << it->second << std::endl;
  // }
  }
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  while (readWord(in, word)) {
    add(word);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);
    }
  }
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

// Assert nsequences_ = sequences_.size()
void Dictionary::threshold(int64_t t) {
  sort(sequences_.begin(), sequences_.end(), [](const entry& e1, const entry& e2) {
      return e1.count > e2.count;
    });
  sequences_.erase(remove_if(sequences_.begin(), sequences_.end(), [&](const entry& e) {
        return e.count < t;
      }), sequences_.end());
  sequences_.shrink_to_fit();
  size_ = 0;
  nsequences_ = 0;
  nlabels_ = 0;
  for (auto it = sequences_.begin(); it != sequences_.end(); ++it) {
    nsequences_++;
    // int32_t h = find(it->word);
    // word2int_[h] = size_++;
    // if (it->type == entry_type::word) nwords_++;
    // if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.count > e2.count;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return (e.type == entry_type::word && e.count < t) ||
               (e.type == entry_type::label && e.count < tl);
      }), words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  // for (size_t i = 0; i < size_; i++) {
  //   real f = real(words_[i].count) / real(ntokens_);
  //   pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  // }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  // for (auto& w : words_) {
  //   if (w.type == type) counts.push_back(w.count);
  // }
  return counts;
}

void Dictionary::addWordNgrams(std::vector<int32_t>& line,
                               const std::vector<int32_t>& hashes,
                               int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(std::vector<int32_t>& line,
                             const std::string& token,
                             int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    // FIXME use utils::seek
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) continue;

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) break;
  }
  return ntokens;
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& ngrams,
                            std::vector<int32_t>& labels) const {
  std::string label;
  std::vector<int32_t> ngrams_comp;

  reset(in);
  ngrams.clear();
  labels.clear();
  readSequence(in, ngrams, ngrams_comp, 300);
  std::getline(in, label);
  auto it = label2int_.find(label.substr(9));
  if (it != label2int_.end()) {
    labels.push_back(it->second);
  }
  return 0;
}

// int32_t Dictionary::getLine(std::istream& in,
//                             std::vector<int32_t>& words,
//                             std::vector<int32_t>& labels) const {
//   std::vector<int32_t> word_hashes;
//   std::string token;
//   int32_t ntokens = 0;

//   reset(in);
//   words.clear();
//   labels.clear();
//   while (readWord(in, token)) {
//     uint32_t h = hash(token);
//     int32_t wid = getId(token, h);
//     entry_type type = wid < 0 ? getType(token) : getType(wid);

//     ntokens++;
//     if (type == entry_type::word) {
//       addSubwords(words, token, wid);
//       word_hashes.push_back(h);
//     } else if (type == entry_type::label && wid >= 0) {
//       labels.push_back(wid - nwords_);
//     }
//     if (token == EOS) break;
//   }
//   addWordNgrams(words, word_hashes, args_->wordNgrams);
//   return ntokens;
// }

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) return;
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  // Reverse lookup
  for (auto it=label2int_.begin(); it!=label2int_.end(); ++it) {
    if (it->second == lid) {
      return it->first;
    }
  }
  throw std::invalid_argument("Could not find label " + std::to_string(lid));
}

void Dictionary::saveString(std::ostream& out, const std::string& s) const {
  out.write(s.data(), s.size() * sizeof(char));
  out.put(0);
}

void Dictionary::loadString(std::istream& in, std::string& s) const {
  char c;
  s.clear();
  while ((c = in.get()) != 0) {
    s.push_back(c);
  }
}

void Dictionary::save(std::ostream& out) const {
  int32_t name2labelsize_ = name2label_.size();
  out.write((char*) &nsequences_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &name2labelsize_, sizeof(int32_t));
  for (int32_t i = 0; i < nsequences_; i++) {
    entry e = sequences_[i];
    saveString(out, e.label);
    saveString(out, e.name);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.seq_pos), sizeof(std::streampos));
    out.write((char*) &(e.name_pos), sizeof(std::streampos));
  }
  for (const auto pair : name2label_) {
    saveString(out, pair.first);
    saveString(out, pair.second);
  }
  for (const auto pair : label2int_) {
    saveString(out, pair.first);
    out.write((char*) &(pair.second), sizeof(int32_t));
  }
}

void Dictionary::load(std::istream& in) {
  sequences_.clear();
  //FIXME
  int32_t name2labelsize_;
  in.read((char*) &nsequences_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &name2labelsize_, sizeof(int32_t));
  for (int32_t i = 0; i < nsequences_; i++) {
    entry e;
    loadString(in, e.label);
    loadString(in, e.name);
    in.read((char*) &(e.count), sizeof(int64_t));
    in.read((char*) &(e.seq_pos), sizeof(std::streampos));
    in.read((char*) &(e.name_pos), sizeof(std::streampos));
    sequences_.push_back(e);
  }
  // FIXME
  name2label_.clear();
  for (int32_t i = 0; i < name2labelsize_; i++) {
    std::string name;
    std::string label;
    loadString(in, name);
    loadString(in, label);
    name2label_[name] = label;
  }
  label2int_.clear();
  for (int32_t i = 0; i < nlabels_; i++) {
    int32_t index;
    std::string label;
    loadString(in, label);
    in.read((char*) &index, sizeof(int32_t));
    label2int_[label] = index;
  }
  // initTableDiscard();
  // initNgrams();

  // int32_t word2intsize = std::ceil(size_ / 0.7);
  // word2int_.assign(word2intsize, -1);
  // for (int32_t i = 0; i < size_; i++) {
  //   word2int_[find(words_[i].word)] = i;
  // }
}

void Dictionary::loadLabelMap() {
  name2label_.clear();
  if (args_->labels.size() != 0) {
    std::ifstream ifs(args_->labels);
    std::string name, label, strain;
    if (!ifs.is_open()) {
      throw std::invalid_argument(args_->labels + " cannot be opened for loading!");
    }
    while (ifs >> name >> strain >> label) {
      name2label_[name] = label;
    }
    ifs.close();
  }
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {words.push_back(*it);}
    else {ngrams.push_back(*it);}
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  // for (int32_t i = 0; i < words_.size(); i++) {
  //   if (getType(i) == entry_type::label || (j < words.size() && words[j] == i)) {
  //     words_[j] = words_[i];
  //     word2int_[find(words_[j].word)] = j;
  //     j++;
  //   }
  // }
  nwords_ = words.size();
  size_ = nwords_ +  nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  // for (auto it : words_) {
  //   std::string entryType = "word";
  //   if (it.type == entry_type::label) {
  //     entryType = "label";
  //   }
  //   out << it.word << " " << it.count << " " << entryType << std::endl;
  // }
}

}
