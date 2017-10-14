#include <gflags/gflags.h>
#include <iostream>
#include <set>
#include <vector>

#include "proto/io.h"
#include "proto/training_dataset.pb.h"
#include "utils/base.h"

using namespace std;

DEFINE_string(input_dictionary, "", "");
DEFINE_int32(min_count, 0, "");
DEFINE_string(output_dictionary, "", "");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  vector<DictionaryEntry> dictionary =
      ReadMessages<DictionaryEntry>(FLAGS_input_dictionary);
  auto out = OpenForWriting(FLAGS_output_dictionary);
  int num_features = 0;
  set<string> feature_types;
  for (const DictionaryEntry& entry : dictionary) {
    if (entry.count() >= FLAGS_min_count) {
      WriteDelimitedTo(entry, out.get());
      ++num_features;
      int sep_index = entry.name().find(" =");
      CHECK(sep_index != -1);
      feature_types.insert(entry.name().substr(0, sep_index));
    }
  }
  cout << "Using " << num_features << " features." << endl;
  cout << endl << "Feature types:" << endl << endl;
  for (const auto& feature_type : feature_types) {
    cout << feature_type << endl;
  }
}
