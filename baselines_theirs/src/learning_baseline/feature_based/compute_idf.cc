#include <boost/functional/hash.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <gflags/gflags.h>

#include "proto/dataset.pb.h"
#include "proto/io.h"
#include "proto/training_dataset.pb.h"

using namespace std;
using edu::stanford::nlp::pipeline::Sentence;
using edu::stanford::nlp::pipeline::Token;

DEFINE_string(input, "", "Path to the input articles.");
DEFINE_string(output_unigrams, "", "Where to output the unigram frequencies.");
DEFINE_string(output_bigrams, "", "Where to output the bigram frequencies.");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  int num_paragraphs = 0;
  unordered_map<string, int> unigram_paragraph_counts;
  unordered_map<pair<string, string>, int, boost::hash<pair<string, string>>>
      unigram_paragraph_coocurrence_counts;
  unordered_map<pair<string, string>, int, boost::hash<pair<string, string>>>
      bigram_paragraph_counts;
  for (const Article& article : ReadMessages<Article>(FLAGS_input)) {
    cout << article.title() << endl;
    for (const Paragraph& paragraph : article.paragraphs()) {
      ++num_paragraphs;

      set<string> unigrams;
      set<pair<string, string>> bigrams;
      for (const Sentence& sentence : paragraph.context().sentence()) {
        for (const Token& token : sentence.token()) {
          unigrams.insert(token.lemma());
        }
        for (int i = 0; i < sentence.token_size() - 1; ++i) {
          bigrams.emplace(sentence.token(i).lemma(),
                          sentence.token(i + 1).lemma());
        }
      }

      for (const string& unigram : unigrams) {
        ++unigram_paragraph_counts[unigram];
        for (const string& unigram2 : unigrams) {
          // This isn't the most optimal handling of bigrams containing two of
          // the same word. Oh well...
          ++unigram_paragraph_coocurrence_counts[make_pair(unigram, unigram2)];
        }
      }
      for (const pair<string, string>& bigram : bigrams) {
        ++bigram_paragraph_counts[bigram];
      }
    }
  }

  unique_ptr<google::protobuf::io::ZeroCopyOutputStream> unigram_output =
      OpenForWriting(FLAGS_output_unigrams);
  for (const auto& unigram_paragraph_count : unigram_paragraph_counts) {
    UnigramWeight weight;
    weight.set_lemma(unigram_paragraph_count.first);
    weight.set_weight(
        log(1.0 * num_paragraphs / unigram_paragraph_count.second));
    WriteDelimitedTo(weight, unigram_output.get());
  }

  // We use the N-gram IDF as defined in
  // http://www.www2015.it/documents/proceedings/proceedings/p960.pdf
  unique_ptr<google::protobuf::io::ZeroCopyOutputStream> bigram_output =
      OpenForWriting(FLAGS_output_bigrams);
  for (const auto& bigram_paragraph_count : bigram_paragraph_counts) {
    BigramWeight weight;
    weight.set_lemma1(bigram_paragraph_count.first.first);
    weight.set_lemma2(bigram_paragraph_count.first.second);
    weight.set_weight(log(
        1.0 * num_paragraphs * bigram_paragraph_count.second /
        pow(unigram_paragraph_coocurrence_counts[bigram_paragraph_count.first],
            2)));
    WriteDelimitedTo(weight, bigram_output.get());
  }
}
