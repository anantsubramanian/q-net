// Given a list of numeric features, bins these into equal-size (histogram)
// buckets based on a TrainingArticle dataset and outputs a configuration file
// containing these buckets.

#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <set>
#include <vector>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <gflags/gflags.h>

#include "proto/io.h"
#include "proto/training_dataset.pb.h"
#include "utils/base.h"

using namespace std;
using google::protobuf::RepeatedPtrField;

DEFINE_string(input, "", "Input of TrainingArticles containing features.");
DEFINE_string(input_dictionary, "", "Input feature dictionary.");
DEFINE_bool(paragraph_level, false,
            "Whether to consider candidate answers at the paragraph level or "
            "at the article level.");
DEFINE_string(features, "",
              "Comma-separated list of features to compute histogram buckets "
              "for.");
DEFINE_int32(buckets, 10, "Number of histogram buckets to use.");
DEFINE_int32(negatives_per_positive, 3,
             "Number of incorrect answers to use for each correct answer.");
DEFINE_string(output_config, "", "Where to output the BucketizerConfig proto.");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(!FLAGS_input.empty());
  CHECK(!FLAGS_input_dictionary.empty());
  CHECK(!FLAGS_features.empty());
  CHECK(!FLAGS_output_config.empty());

  srand(123);

  set<string> features;
  boost::split(features, FLAGS_features, boost::is_any_of(","));

  map<int, string> feature_names;
  map<int, vector<float>> feature_values;
  vector<DictionaryEntry> dictionary =
      ReadMessages<DictionaryEntry>(FLAGS_input_dictionary);
  for (const auto& entry : dictionary) {
    if (features.count(entry.name())) {
      feature_names.emplace(entry.index(), entry.name());
      feature_values.emplace(entry.index(), vector<float>());
    }
  }

  CHECK(features.size() == feature_values.size());

  std::unique_ptr<google::protobuf::io::ZeroCopyInputStream> input =
      OpenForReading(FLAGS_input);
  while (true) {
    TrainingArticle article;
    if (!ReadDelimitedFrom(input.get(), &article)) {
      break;
    }

    auto process_question = [&feature_values](
        const TrainingQuestionAnswer& qa) {
      vector<int> candidate_answer_indices;
      candidate_answer_indices.push_back(qa.correctanswerindex());
      for (int i = 0;
           i < FLAGS_negatives_per_positive &&
           candidate_answer_indices.size() < qa.candidateanswerfeatures_size();
           ++i) {
        int times = 0;
        while (true) {
          int negative_index = rand() % qa.candidateanswerfeatures_size();
          if (find(candidate_answer_indices.begin(),
                   candidate_answer_indices.end(),
                   negative_index) == candidate_answer_indices.end()) {
            candidate_answer_indices.push_back(negative_index);
            break;
          }
          ++times;
        }
      }
      for (int index : candidate_answer_indices) {
        const CandidateAnswerFeatures& features =
            qa.candidateanswerfeatures(index);
        for (int i = 0; i < features.indices_size(); ++i) {
          auto it = feature_values.find(features.indices(i));
          if (it != feature_values.end()) {
            it->second.push_back(features.values(i));
          }
        }
      }
    };

    if (FLAGS_paragraph_level) {
      for (const auto& paragraph : article.paragraphs()) {
        for (const auto& question : paragraph.questions()) {
          process_question(question);
        }
      }
    } else {
      for (const auto& question : article.questions()) {
        process_question(question);
      }
    }

    for (const auto& question : article.questions()) {
      for (const auto& features : question.candidateanswerfeatures()) {
        for (int i = 0; i < features.indices_size(); ++i) {
          auto it = feature_values.find(features.indices(i));
          if (it != feature_values.end()) {
            it->second.push_back(features.values(i));
          }
        }
      }
    }
  }

  BucketizerConfig config;
  for (auto& feature : feature_values) {
    auto* output_feature = config.add_features();
    output_feature->set_name(feature_names[feature.first]);
    sort(feature.second.begin(), feature.second.end());

    const vector<float>& values = feature.second;
    output_feature->add_lower_bounds(values[0]);
    for (int bucket = 1; bucket < FLAGS_buckets; ++bucket) {
      int start = upper_bound(values.begin(), values.end(),
                              output_feature->lower_bounds(bucket - 1)) -
                  values.begin();
      if (start == values.size()) {
        break;
      }

      output_feature->add_lower_bounds(
          values[(values.size() - start) / (FLAGS_buckets - bucket + 1) +
                 start]);
    }

    cout << output_feature->name() << endl;
    for (float lower_bound : output_feature->lower_bounds()) {
      cout << lower_bound << endl;
    }
  }
  unique_ptr<google::protobuf::io::ZeroCopyOutputStream> output =
      OpenForWriting(FLAGS_output_config);
  CHECK(WriteDelimitedTo(config, output.get()));
}
