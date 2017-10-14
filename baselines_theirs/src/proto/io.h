#ifndef SRC_PROTO_IO_H_
#define SRC_PROTO_IO_H_

#include <memory>
#include <string>
#include <vector>

#include <google/protobuf/message_lite.h>
#include <google/protobuf/io/zero_copy_stream.h>

std::unique_ptr<google::protobuf::io::ZeroCopyOutputStream> OpenForWriting(
    const std::string& path);

bool WriteDelimitedTo(const google::protobuf::MessageLite& message,
                      google::protobuf::io::ZeroCopyOutputStream* rawOutput);

std::unique_ptr<google::protobuf::io::ZeroCopyInputStream> OpenForReading(
    const std::string& path);

bool ReadDelimitedFrom(google::protobuf::io::ZeroCopyInputStream* rawInput,
                       google::protobuf::MessageLite* message);

template <typename MessageType>
std::vector<MessageType> ReadMessages(const std::string& path, int num = -1) {
  std::vector<MessageType> result;
  std::unique_ptr<google::protobuf::io::ZeroCopyInputStream> input =
      OpenForReading(path);
  while (num == -1 || result.size() < num) {
    result.emplace_back();
    if (!ReadDelimitedFrom(input.get(), &result.back())) {
      result.pop_back();
      break;
    }
  }
  return result;
}

#endif /* SRC_PROTO_IO_H_ */
