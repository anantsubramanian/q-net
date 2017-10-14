CXX = g++
CXXFLAGS = -Wall -std=c++11 -O2 -Isrc -Wno-sign-compare -pthread
LDFLAGS=-lprotobuf -lgflags -lboost_system -lboost_thread

#.SILENT:

bin/learning_baseline/candidate_extraction: build/proto/CoreNLP.pb.o build/proto/dataset.pb.o build/proto/training_dataset.pb.o build/proto/io.o build/utils/thread_pool.o build/learning_baseline/feature_based/feature_extractor.o build/learning_baseline/candidate_extraction.o
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

bin/learning_baseline/feature_based/bucketizer: build/proto/training_dataset.pb.o build/proto/io.o build/learning_baseline/feature_based/bucketizer.o
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	
bin/learning_baseline/feature_based/compute_idf: build/proto/CoreNLP.pb.o build/proto/dataset.pb.o build/proto/training_dataset.pb.o build/proto/io.o build/learning_baseline/feature_based/compute_idf.o
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	
bin/learning_baseline/feature_based/filter_dictionary: build/proto/training_dataset.pb.o build/proto/io.o build/learning_baseline/feature_based/filter_dictionary.o
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

build/learning_baseline/candidate_extraction.o: src/learning_baseline/candidate_extraction.cc src/utils/base.h src/proto/dataset.pb.h src/proto/training_dataset.pb.h src/proto/io.h src/utils/thread_pool.h src/learning_baseline/feature_based/feature_extractor.h 
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@
	
build/learning_baseline/feature_based/bucketizer.o: src/learning_baseline/feature_based/bucketizer.cc src/utils/base.h src/proto/training_dataset.pb.h src/proto/io.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@
	
build/learning_baseline/feature_based/feature_extractor.o: src/learning_baseline/feature_based/feature_extractor.cc src/learning_baseline/feature_based/feature_extractor.h src/utils/base.h src/proto/CoreNLP.pb.h src/proto/dataset.pb.h src/proto/training_dataset.pb.h src/proto/io.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@
	
build/learning_baseline/feature_based/compute_idf.o: src/learning_baseline/feature_based/compute_idf.cc src/proto/dataset.pb.h src/proto/io.h src/proto/training_dataset.pb.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@

build/learning_baseline/feature_based/filter_dictionary.o: src/learning_baseline/feature_based/filter_dictionary.cc src/proto/io.h src/proto/training_dataset.pb.h src/utils/base.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@

build/proto/CoreNLP.pb.o: src/proto/CoreNLP.pb.cc src/proto/CoreNLP.pb.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@

build/proto/dataset.pb.o: src/proto/dataset.pb.cc src/proto/dataset.pb.h src/proto/CoreNLP.pb.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@

build/proto/training_dataset.pb.o: src/proto/training_dataset.pb.cc src/proto/training_dataset.pb.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@
	
build/proto/io.o: src/proto/io.cc src/proto/io.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@

build/utils/thread_pool.o: src/utils/thread_pool.cc src/utils/thread_pool.h
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf build bin
