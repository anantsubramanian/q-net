#ifndef SRC_LEARNING_BASELINE_FEATURE_BASED_FEATURE_EXTRACTOR_H_
#define SRC_LEARNING_BASELINE_FEATURE_BASED_FEATURE_EXTRACTOR_H_

#include <boost/atomic.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "proto/CoreNLP.pb.h"
#include "proto/dataset.pb.h"
#include "proto/training_dataset.pb.h"

class NgramWeights {
 public:
  NgramWeights();

  float GetUnigramWeight(const std::string& lemma) const;
  float GetBigramWeight(const std::string& lemma1,
                        const std::string& lemma2) const;

 private:
  std::unordered_map<std::string, float> unigram_weights_;
  float max_unigram_weight_;
  std::unordered_map<std::pair<std::string, std::string>, float,
                     boost::hash<std::pair<std::string, std::string>>>
      bigram_weights_;
  float max_bigram_weight_;
};

class Bucketizer {
 public:
  Bucketizer(const std::string& config_file);

  const std::string* Bucketize(const std::string* feature, float value) const;

 private:
  std::map<std::string, std::pair<std::vector<std::string>, std::vector<float>>>
      names_and_lower_bounds_;
};

// Thread-safe feature dictionary.
class FeatureDictionary {
 public:
  FeatureDictionary();

  int Count(const std::string& feature);

  void Save() const;

 private:
  bool read_only_;
  boost::shared_mutex lock_;
  std::unordered_map<std::string, int> dictionary_;
  std::unordered_map<int, boost::atomic_int> count_;
};

class FeatureExtractor {
 public:
  struct PreprocessedSentence {
    std::map<std::string, int> lemma_pos;
    std::vector<std::vector<const std::string*>> dep_edges;
    std::vector<std::vector<int>> shortest_dep_path;
    std::vector<std::vector<int>> next_in_dep_path;
  };

  FeatureExtractor();

  void Save();

  void PreprocessSentences(
      const Article& article,
      std::vector<std::vector<FeatureExtractor::PreprocessedSentence>>*
          preprocessed_sentences) const;

  void ExtractFeatures(
      const Article& article,
      const std::vector<std::vector<FeatureExtractor::PreprocessedSentence>>&
          preprocessed_sentences,
      const google::protobuf::RepeatedPtrField<CandidateAnswer>&
          candidate_answers,
      const std::vector<const edu::stanford::nlp::pipeline::ParseTree*>&
          parse_trees,
      const edu::stanford::nlp::pipeline::Document& question,
      TrainingQuestionAnswer* training_qa);

 private:
  struct PreprocessedQuestion {
    std::map<std::string, float> question_unigram_weights;
    std::map<std::string, std::map<std::string, float>> question_bigram_weights;
    // Just the first sentence of the question.
    PreprocessedSentence preprocessed_sentence;
  };

  void PreprocessSentence(
      const edu::stanford::nlp::pipeline::Sentence& sentence,
      PreprocessedSentence* out) const;

  std::unique_ptr<PreprocessedQuestion> PreprocessQuestion(
      const edu::stanford::nlp::pipeline::Document& question) const;

  void ExtractFeatures(
      const edu::stanford::nlp::pipeline::Document& question,
      const PreprocessedQuestion& qa,
      const edu::stanford::nlp::pipeline::Sentence& sentence,
      const PreprocessedSentence& preprocessed_sentence,
      const edu::stanford::nlp::pipeline::ParseTree& parse_tree,
      int span_begin_index, int span_length, CandidateAnswerFeatures* features);

  void AddFeature(const std::string& feature, float value,
                  CandidateAnswerFeatures* features);

  void AddUnigramTfIfFeature(
      const std::string& feature,
      const edu::stanford::nlp::pipeline::Sentence& sentence, int begin_index,
      int length, CandidateAnswerFeatures* features);
  void AddUnigramMatchTfIfFeature(
      const std::string& feature, const PreprocessedQuestion& qa,
      const edu::stanford::nlp::pipeline::Sentence& sentence, int begin_index,
      int length, CandidateAnswerFeatures* features);
  void AddBigramMatchTfIfFeature(
      const std::string& feature, const PreprocessedQuestion& qa,
      const edu::stanford::nlp::pipeline::Sentence& sentence, int begin_index,
      int length, CandidateAnswerFeatures* features);

  void AddDepPathFeatures(
      const edu::stanford::nlp::pipeline::Sentence& sentence,
      const PreprocessedSentence& preprocessed_sentence,
      const edu::stanford::nlp::pipeline::Document& question,
      const PreprocessedQuestion& preprocessed_question, int wh_token_index,
      const std::string& wh_token_string, int span_begin_index, int span_length,
      CandidateAnswerFeatures* features);
  std::string ConstructDepPath(
      const edu::stanford::nlp::pipeline::Sentence& sentence,
      const PreprocessedSentence& sen, int from, int to,
      bool include_lemmas) const;

  const NgramWeights ngram_weights_;
  std::unique_ptr<const Bucketizer> bucketizer_;
  FeatureDictionary dictionary_;
  std::set<std::string> uninteresting_pos_;
  std::set<std::string> wh_pos_;
};

#endif /* SRC_LEARNING_BASELINE_FEATURE_BASED_FEATURE_EXTRACTOR_H_ */
