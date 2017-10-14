#include "feature_extractor.h"

#include <algorithm>
#include <fcntl.h>
#include <memory>
#include <utility>
#include <sstream>

#include <boost/thread/pthread/shared_mutex.hpp>
#include <gflags/gflags.h>
#include <google/protobuf/message_lite.h>

#include "proto/io.h"
#include "utils/base.h"

using namespace std;
using edu::stanford::nlp::pipeline::Document;
using edu::stanford::nlp::pipeline::ParseTree;
using edu::stanford::nlp::pipeline::Sentence;
using google::protobuf::RepeatedPtrField;

DEFINE_string(unigram_weights, "", "Path to the unigram weights file.");
DEFINE_double(
    dep_lemma_max_weight, 4.0,
    "Maximum unigram weight of a lemma used in a dependency path features.");
DEFINE_int32(max_dep_path_length, 4,
             "Maximum length of a depedency path to consider.");
DEFINE_string(bigram_weights, "", "Path to the bigram weights file.");
DEFINE_string(bucketizer_config, "", "Path to the bucketizer config.");
DEFINE_string(input_dictionary, "", "Path to the dictionary to read.");
DEFINE_bool(only_numeric_features, false,
            "Whether to extract only numeric features for bucketization.");
DEFINE_string(output_dictionary, "",
              "Path where to output the feature dictionary.");

NgramWeights::NgramWeights() {
  CHECK(!FLAGS_unigram_weights.empty());
  CHECK(!FLAGS_bigram_weights.empty());

  max_unigram_weight_ = std::numeric_limits<float>::min();
  for (const auto& weight :
       ReadMessages<UnigramWeight>(FLAGS_unigram_weights)) {
    unigram_weights_[weight.lemma()] = weight.weight();
    max_unigram_weight_ = max(max_unigram_weight_, weight.weight());
  }

  max_bigram_weight_ = std::numeric_limits<float>::min();
  for (const auto& weight : ReadMessages<BigramWeight>(FLAGS_bigram_weights)) {
    bigram_weights_[make_pair(weight.lemma1(), weight.lemma2())] =
        weight.weight();
    max_bigram_weight_ = max(max_bigram_weight_, weight.weight());
  }
}

float NgramWeights::GetUnigramWeight(const std::string& lemma) const {
  auto it = unigram_weights_.find(lemma);
  if (it != unigram_weights_.end()) {
    return it->second;
  }
  return max_unigram_weight_;
}

float NgramWeights::GetBigramWeight(const std::string& lemma1,
                                    const std::string& lemma2) const {
  auto it = bigram_weights_.find(make_pair(lemma1, lemma2));
  if (it != bigram_weights_.end()) {
    return it->second;
  }
  return max_bigram_weight_;
}

Bucketizer::Bucketizer(const string& config_file) {
  vector<BucketizerConfig> configs =
      ReadMessages<BucketizerConfig>(config_file);
  CHECK(configs.size() == 1);
  for (const auto& feature : configs[0].features()) {
    vector<string> names;
    vector<float> lower_bounds;
    for (int bucket = 0; bucket < feature.lower_bounds_size(); ++bucket) {
      ostringstream sout;
      sout << feature.name() << " Bucket " << bucket << " of "
           << feature.lower_bounds_size() << ": "
           << feature.lower_bounds(bucket) << " - ";
      if (bucket + 1 < feature.lower_bounds_size()) {
        sout << feature.lower_bounds(bucket + 1);
      } else {
        sout << "inf";
      }
      names.emplace_back(sout.str());
      lower_bounds.emplace_back(feature.lower_bounds(bucket));
    }
    names_and_lower_bounds_.emplace(feature.name(),
                                    make_pair(names, lower_bounds));
  }
}

const string* Bucketizer::Bucketize(const string* feature, float value) const {
  auto it = names_and_lower_bounds_.find(*feature);
  if (it == names_and_lower_bounds_.end()) {
    return nullptr;
  }

  const vector<string>& names = it->second.first;
  const vector<float>& lower_bounds = it->second.second;
  int bucket =
      max<int>(upper_bound(lower_bounds.begin(), lower_bounds.end(), value) -
                   lower_bounds.begin() - 1,
               0);

  return &names[bucket];
}

FeatureDictionary::FeatureDictionary() {
  CHECK(!FLAGS_input_dictionary.empty() || !FLAGS_output_dictionary.empty());
  if (!FLAGS_input_dictionary.empty()) {
    read_only_ = true;
    vector<DictionaryEntry> entries =
        ReadMessages<DictionaryEntry>(FLAGS_input_dictionary);
    for (const DictionaryEntry& entry : entries) {
      dictionary_[entry.name()] = entry.index();
    }
  } else {
    read_only_ = false;
  }
}

int FeatureDictionary::Count(const string& feature) {
  if (read_only_) {
    auto it = dictionary_.find(feature);
    if (it == dictionary_.end()) {
      return -1;
    }
    return it->second;
  } else {
    lock_.lock_shared();
    auto it = dictionary_.find(feature);
    if (it != dictionary_.end()) {
      int num = it->second;
      ++count_[num];
      lock_.unlock_shared();
      return num;
    }

    lock_.unlock_shared();
    lock_.lock();

    // Ensure that some other writer hasn't added it already when we unlocked
    // above.
    it = dictionary_.find(feature);
    if (it != dictionary_.end()) {
      int num = it->second;

      ++count_[num];
      lock_.unlock();
      return num;
    }

    int num = dictionary_.size();
    dictionary_[feature] = num;
    count_[num] = 1;
    lock_.unlock();
    if (num % 1000 == 0) {
      cout << num << " " << feature << endl;
    }
    return num;
  }
}

void FeatureDictionary::Save() const {
  if (!FLAGS_output_dictionary.empty()) {
    // Sort by index.
    vector<pair<int, string>> sorted_entries;
    for (const auto& name_index_pair : dictionary_) {
      sorted_entries.emplace_back(name_index_pair.second,
                                  name_index_pair.first);
    }
    sort(sorted_entries.begin(), sorted_entries.end());

    unique_ptr<google::protobuf::io::ZeroCopyOutputStream> output =
        OpenForWriting(FLAGS_output_dictionary);
    for (const auto& index_name_pair : sorted_entries) {
      DictionaryEntry entry;
      entry.set_name(index_name_pair.second);
      entry.set_index(index_name_pair.first);
      entry.set_count(count_.find(index_name_pair.first)->second);
      CHECK(WriteDelimitedTo(entry, output.get()));
    }
  }
}

FeatureExtractor::FeatureExtractor() {
  if (!FLAGS_bucketizer_config.empty()) {
    bucketizer_.reset(new Bucketizer(FLAGS_bucketizer_config));
  }
  for (const char* pos :
       {"CC", ".", ",", ":", "-LRB-", "-RRB-", "POS", "DT", "IN", "TO"}) {
    uninteresting_pos_.insert(pos);
  }

  for (const char* pos : {"WRB", "WP", "WDT", "WP$"}) {
    wh_pos_.insert(pos);
  }
}

void FeatureExtractor::Save() { dictionary_.Save(); }

void FeatureExtractor::ExtractFeatures(
    const Article& article,
    const vector<vector<FeatureExtractor::PreprocessedSentence>>&
        preprocessed_sentences,
    const RepeatedPtrField<CandidateAnswer>& candidate_answers,
    const vector<const ParseTree*>& parse_trees, const Document& question,
    TrainingQuestionAnswer* training_qa) {
  unique_ptr<PreprocessedQuestion> preprocessed_question =
      PreprocessQuestion(question);
  for (int i = 0; i < candidate_answers.size(); ++i) {
    const CandidateAnswer& candidate_answer = candidate_answers.Get(i);
    const Paragraph& paragraph =
        article.paragraphs(candidate_answer.paragraphindex());
    const Sentence& sentence =
        paragraph.context().sentence(candidate_answer.sentenceindex());
    const PreprocessedSentence& preprocessed_sentence = preprocessed_sentences
        [candidate_answer.paragraphindex()][candidate_answer.sentenceindex()];

    ExtractFeatures(question, *preprocessed_question, sentence,
                    preprocessed_sentence, *parse_trees[i],
                    candidate_answer.spanbeginindex(),
                    candidate_answer.spanlength(),
                    training_qa->add_candidateanswerfeatures());
  }
}

void FeatureExtractor::PreprocessSentences(
    const Article& article,
    vector<vector<FeatureExtractor::PreprocessedSentence>>*
        preprocessed_sentences) const {
  for (const Paragraph& paragraph : article.paragraphs()) {
    preprocessed_sentences->emplace_back(paragraph.context().sentence_size());
    for (int i = 0; i < paragraph.context().sentence_size(); ++i) {
      const Sentence& sentence = paragraph.context().sentence(i);
      PreprocessedSentence* preprocessed_sentence =
          &preprocessed_sentences->back()[i];
      PreprocessSentence(sentence, preprocessed_sentence);
    }
  }
}

void FeatureExtractor::PreprocessSentence(
    const edu::stanford::nlp::pipeline::Sentence& sentence,
    PreprocessedSentence* out) const {
  for (int i = 0; i < sentence.token_size(); ++i) {
    out->lemma_pos[sentence.token(i).lemma()] = i;
  }

  for (int i = 0; i < sentence.token_size(); ++i) {
    out->dep_edges.emplace_back(sentence.token_size(), nullptr);
    out->shortest_dep_path.emplace_back(sentence.token_size(),
                                        std::numeric_limits<int>::max());
    out->next_in_dep_path.emplace_back(sentence.token_size(), -1);
  }
  for (const auto& edge : sentence.basicdependencies().edge()) {
    out->dep_edges[edge.source() - 1][edge.target() - 1] = &edge.dep();
    out->shortest_dep_path[edge.source() - 1][edge.target() - 1] = 1;
    out->shortest_dep_path[edge.target() - 1][edge.source() - 1] = 1;
    out->next_in_dep_path[edge.source() - 1][edge.target() - 1] =
        edge.target() - 1;
    out->next_in_dep_path[edge.target() - 1][edge.source() - 1] =
        edge.source() - 1;
  }
  // Floyd-Warshall:
  for (int k = 0; k < sentence.token_size(); ++k) {
    for (int i = 0; i < sentence.token_size(); ++i) {
      if (i == k) {
        continue;
      }
      for (int j = 0; j < sentence.token_size(); ++j) {
        if (k == j) {
          continue;
        }

        if (out->shortest_dep_path[i][k] != std::numeric_limits<int>::max() &&
            out->shortest_dep_path[k][j] != std::numeric_limits<int>::max() &&
            out->shortest_dep_path[i][k] + out->shortest_dep_path[k][j] <
                out->shortest_dep_path[i][j]) {
          out->shortest_dep_path[i][j] =
              out->shortest_dep_path[i][k] + out->shortest_dep_path[k][j];
          out->next_in_dep_path[i][j] = out->next_in_dep_path[i][k];
        }
      }
    }
  }
}

std::unique_ptr<FeatureExtractor::PreprocessedQuestion>
FeatureExtractor::PreprocessQuestion(const Document& question) const {
  std::unique_ptr<PreprocessedQuestion> res(new PreprocessedQuestion);
  for (const Sentence& sentence : question.sentence()) {
    for (int i = 0; i < sentence.token_size(); ++i) {
      res->question_unigram_weights[sentence.token(i).lemma()] =
          ngram_weights_.GetUnigramWeight(sentence.token(i).lemma());
      if (i + 1 < sentence.token_size()) {
        res->question_bigram_weights[sentence.token(i)
                                         .lemma()][sentence.token(i + 1)
                                                       .lemma()] =
            ngram_weights_.GetBigramWeight(sentence.token(i).lemma(),
                                           sentence.token(i + 1).lemma());
      }
    }
  }

  PreprocessSentence(question.sentence(0), &res->preprocessed_sentence);

  return res;
}

void FeatureExtractor::ExtractFeatures(
    const Document& question, const PreprocessedQuestion& preprocessed_question,
    const Sentence& sentence, const PreprocessedSentence& preprocessed_sentence,
    const ParseTree& parse_tree, int span_begin_index, int span_length,
    CandidateAnswerFeatures* features) {
  AddFeature("Left Length =", span_begin_index, features);
  AddFeature("Right Length =",
             sentence.token_size() - (span_begin_index + span_length),
             features);
  AddFeature("Sentence Length =", sentence.token_size(), features);
  AddFeature("Span Length =", span_length, features);

  AddUnigramTfIfFeature("Unigram TF IF (Span) =", sentence, span_begin_index,
                        span_length, features);

  AddUnigramMatchTfIfFeature("Unigram QA Match TF IF (Left of Span) =",
                             preprocessed_question, sentence, 0,
                             span_begin_index, features);
  AddUnigramMatchTfIfFeature(
      "Unigram QA Match TF IF (Right of Span) =", preprocessed_question,
      sentence, span_begin_index + span_length,
      sentence.token_size() - (span_begin_index + span_length), features);
  AddUnigramMatchTfIfFeature("Unigram QA Match TF IF (Whole Sentence) =",
                             preprocessed_question, sentence, 0,
                             sentence.token_size(), features);
  AddUnigramMatchTfIfFeature("Unigram QA Match TF IF (Span) =",
                             preprocessed_question, sentence, span_begin_index,
                             span_length, features);

  AddBigramMatchTfIfFeature("Bigram QA Match TF IF (Left of Span) =",
                            preprocessed_question, sentence, 0,
                            span_begin_index, features);
  AddBigramMatchTfIfFeature(
      "Bigram QA Match TF IF (Right of Span) =", preprocessed_question,
      sentence, span_begin_index + span_length,
      sentence.token_size() - (span_begin_index + span_length), features);
  AddBigramMatchTfIfFeature("Bigram QA Match TF IF (Whole Sentence) =",
                            preprocessed_question, sentence, 0,
                            sentence.token_size(), features);
  AddBigramMatchTfIfFeature("Bigram QA Match TF IF (Span) =",
                            preprocessed_question, sentence, span_begin_index,
                            span_length, features);

  if (!FLAGS_only_numeric_features) {
    int wh_token_index = -1;
    string wh_token_string;
    for (int i = 0; i < question.sentence(0).token_size(); ++i) {
      const auto& token = question.sentence(0).token(i);
      if (wh_pos_.count(token.pos()) && token.lemma() != "that") {
        wh_token_index = i;
        wh_token_string = token.pos() + " " + token.lemma();
        break;
      }
    }

    AddFeature(string("Parse Tree Label = ") + parse_tree.value(), 1.0,
               features);
    if (wh_token_index != -1) {
      stringstream sout;
      sout << "Wh + Parse Tree Label = " << wh_token_string
           << " " + parse_tree.value();
      AddFeature(sout.str(), 1.0, features);
    }

    string all_pos;
    {
      stringstream sout;
      for (int i = span_begin_index; i < span_begin_index + span_length; ++i) {
        sout << " " << sentence.token(i).pos();
      }
      all_pos = sout.str();
    }
    AddFeature(string("All Pos =") + all_pos, 1.0, features);
    if (wh_token_index != -1) {
      stringstream sout;
      sout << "Wh + All Pos = " << wh_token_string << all_pos;
      AddFeature(sout.str(), 1.0, features);
    }

    const string& sentence_root_lemma =
        sentence.token(sentence.basicdependencies().root(0) - 1).lemma();
    const string& question_root_lemma =
        question.sentence(0)
            .token(question.sentence(0).basicdependencies().root(0) - 1)
            .lemma();
    if (sentence_root_lemma == question_root_lemma) {
      AddFeature("Root Match = 1", 1.0, features);
    }
    if (preprocessed_sentence.lemma_pos.find(question_root_lemma) !=
        preprocessed_sentence.lemma_pos.end()) {
      AddFeature("Sentence Contains Question Root = 1", 1.0, features);
    }
    if (preprocessed_question.preprocessed_sentence.lemma_pos.find(
            sentence_root_lemma) !=
        preprocessed_question.preprocessed_sentence.lemma_pos.end()) {
      AddFeature("Question Contains Sentence Root = 1", 1.0, features);
    }

    AddDepPathFeatures(sentence, preprocessed_sentence, question,
                       preprocessed_question, wh_token_index, wh_token_string,
                       span_begin_index, span_length, features);
  }

  // Sort and unique the features.
  if (bucketizer_ && features->indices_size() > 1) {
    sort(features->mutable_indices()->begin(),
         features->mutable_indices()->end());
    int new_length = 1;
    for (int i = 1; i < features->indices_size(); ++i) {
      if (features->indices(i) != features->indices(new_length - 1)) {
        features->set_indices(new_length, features->indices(i));
        ++new_length;
      }
    }

    features->mutable_indices()->Truncate(new_length);
  }
}

void FeatureExtractor::AddFeature(const std::string& feature, float value,
                                  CandidateAnswerFeatures* features) {
  if (!bucketizer_) {
    int index = dictionary_.Count(feature);
    if (index != -1) {
      features->add_indices(dictionary_.Count(feature));
      features->add_values(value);
    }
  } else {
    const string* bucketized = bucketizer_->Bucketize(&feature, value);
    int index;
    if (bucketized == nullptr) {
      CHECK(value == 1.0);
      index = dictionary_.Count(feature);
    } else {
      index = dictionary_.Count(*bucketized);
    }
    if (index != -1) {
      features->add_indices(index);
    }
  }
}

void FeatureExtractor::AddUnigramTfIfFeature(
    const std::string& feature,
    const edu::stanford::nlp::pipeline::Sentence& sentence, int begin_index,
    int length, CandidateAnswerFeatures* features) {
  float tf_if = 0;
  for (int i = begin_index; i - begin_index < length; ++i) {
    tf_if += ngram_weights_.GetUnigramWeight(sentence.token(i).lemma());
  }
  AddFeature(feature, tf_if, features);
}

void FeatureExtractor::AddUnigramMatchTfIfFeature(
    const std::string& feature, const PreprocessedQuestion& qa,
    const edu::stanford::nlp::pipeline::Sentence& sentence, int begin_index,
    int length, CandidateAnswerFeatures* features) {
  float tf_if = 0;
  for (int i = begin_index; i - begin_index < length; ++i) {
    auto it = qa.question_unigram_weights.find(sentence.token(i).lemma());
    if (it != qa.question_unigram_weights.end()) {
      tf_if += it->second;
    }
  }
  AddFeature(feature, tf_if, features);
}

void FeatureExtractor::AddBigramMatchTfIfFeature(
    const std::string& feature, const PreprocessedQuestion& qa,
    const edu::stanford::nlp::pipeline::Sentence& sentence, int begin_index,
    int length, CandidateAnswerFeatures* features) {
  float tf_if = 0;
  for (int i = begin_index; i - begin_index + 1 < length; ++i) {
    auto it1 = qa.question_bigram_weights.find(sentence.token(i).lemma());
    if (it1 != qa.question_bigram_weights.end()) {
      auto it2 = it1->second.find(sentence.token(i + 1).lemma());
      if (it2 != it1->second.end()) {
        tf_if += it2->second;
      }
    }
  }
  AddFeature(feature, tf_if, features);
}

void FeatureExtractor::AddDepPathFeatures(
    const Sentence& sentence, const PreprocessedSentence& preprocessed_sentence,
    const edu::stanford::nlp::pipeline::Document& question,
    const PreprocessedQuestion& preprocessed_question, int wh_token_index,
    const string& wh_token_string, int span_begin_index, int span_length,
    CandidateAnswerFeatures* features) {
  for (const auto& question_lemma_and_pos :
       preprocessed_question.preprocessed_sentence.lemma_pos) {
    if (uninteresting_pos_.count(
            question.sentence(0).token(question_lemma_and_pos.second).pos())) {
      continue;
    }

    auto sentence_lemma_and_pos =
        preprocessed_sentence.lemma_pos.find(question_lemma_and_pos.first);
    if (sentence_lemma_and_pos != preprocessed_sentence.lemma_pos.end()) {
      // Matching lemma falls inside the answer span.
      if (span_begin_index <= sentence_lemma_and_pos->second &&
          sentence_lemma_and_pos->second < span_begin_index + span_length) {
        continue;
      }

      int distance_to_span = std::numeric_limits<int>::max();
      int closest_token_in_span = -1;
      for (int i = span_begin_index; i < span_begin_index + span_length; ++i) {
        if (uninteresting_pos_.count(sentence.token(i).pos())) {
          continue;
        }

        int shortest_dep_path =
            preprocessed_sentence
                .shortest_dep_path[sentence_lemma_and_pos->second][i];
        if (shortest_dep_path < distance_to_span) {
          distance_to_span = shortest_dep_path;
          closest_token_in_span = i;
        }
      }
      if (closest_token_in_span == -1) {
        // Only path goes through an uninteresting part of speech.
        continue;
      }

      for (bool dep_path_lemma : {false, true}) {
        string wh_path;
        bool wh_path_has_words = false;
        if (wh_token_index != -1) {
          int distance_to_wh =
              preprocessed_question.preprocessed_sentence.shortest_dep_path
                  [wh_token_index][question_lemma_and_pos.second];
          if (distance_to_wh > FLAGS_max_dep_path_length) {
            wh_path = " ... long path ... ";
          } else {
            wh_path = ConstructDepPath(
                question.sentence(0),
                preprocessed_question.preprocessed_sentence, wh_token_index,
                question_lemma_and_pos.second, dep_path_lemma);
            if (distance_to_wh > 1) {
              wh_path_has_words = true;
            }
          }
        }

        string answer_path;
        bool answer_path_has_words = false;
        if (distance_to_span > FLAGS_max_dep_path_length) {
          answer_path = " ... long path ... ";
        } else {
          answer_path = ConstructDepPath(sentence, preprocessed_sentence,
                                         sentence_lemma_and_pos->second,
                                         closest_token_in_span, dep_path_lemma);
          if (distance_to_span > 1) {
            answer_path_has_words = true;
          }
        }

        for (bool include_wh_path : {false, true}) {
          if (include_wh_path && wh_token_index == -1) {
            continue;
          }

          if (dep_path_lemma &&
              !((include_wh_path && wh_path_has_words) ||
                answer_path_has_words)) {
            // In this case this feature will be exactly the same as the feature
            // with dep_path_lemma = false. Skip it.
            continue;
          }

          for (bool sentence_lemma : {false, true}) {
            if (sentence_lemma &&
                ngram_weights_.GetUnigramWeight(
                    sentence.token(sentence_lemma_and_pos->second).lemma()) >
                    FLAGS_dep_lemma_max_weight) {
              continue;
            }

            for (bool answer_lemma : {false, true}) {
              if (answer_lemma &&
                  ngram_weights_.GetUnigramWeight(
                      sentence.token(closest_token_in_span).lemma()) >
                      FLAGS_dep_lemma_max_weight) {
                continue;
              }

              stringstream sout;

              if (include_wh_path) {
                sout << "Wh Path + ";
              }
              if (sentence_lemma) {
                sout << "SL";
              } else {
                sout << "SPOS";
              }
              sout << " + Dep Path + ";
              if (answer_lemma) {
                sout << "AL";
              } else {
                sout << "APOS";
              }
              if (dep_path_lemma) {
                sout << " (Path Lemmas)";
              }
              sout << " = ";

              if (include_wh_path) {
                sout << wh_token_string << wh_path;
              }
              if (sentence_lemma) {
                sout << sentence.token(sentence_lemma_and_pos->second).lemma();
              } else {
                sout << sentence.token(sentence_lemma_and_pos->second).pos();
              }
              sout << answer_path;
              if (answer_lemma) {
                sout << sentence.token(closest_token_in_span).lemma();
              } else {
                sout << sentence.token(closest_token_in_span).pos();
              }

              AddFeature(sout.str(), 1.0, features);
            }
          }
        }
      }
    }
  }
}

string FeatureExtractor::ConstructDepPath(const Sentence& sentence,
                                          const PreprocessedSentence& sen,
                                          int from, int to,
                                          bool include_lemmas) const {
  stringstream sout;
  while (from != to) {
    int next = sen.next_in_dep_path[from][to];
    CHECK(next != -1);
    CHECK(sen.dep_edges[from][next] != nullptr ||
          sen.dep_edges[next][from] != nullptr);
    if (sen.dep_edges[from][next] != nullptr) {
      sout << " - " << *sen.dep_edges[from][next] << " -> ";
    } else {
      sout << " <- " << *sen.dep_edges[next][from] << " - ";
    }
    if (next != to) {
      if (include_lemmas) {
        sout << sentence.token(next).lemma();
      } else {
        sout << sentence.token(next).pos();
      }
    }
    from = next;
  }
  return sout.str();
}
