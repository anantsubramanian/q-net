#include "proto/io.h"

#include <fcntl.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

std::unique_ptr<google::protobuf::io::ZeroCopyOutputStream> OpenForWriting(
    const std::string& path) {
  google::protobuf::io::FileOutputStream* out =
      new google::protobuf::io::FileOutputStream(
          open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644));
  out->SetCloseOnDelete(true);
  return std::unique_ptr<google::protobuf::io::ZeroCopyOutputStream>(out);
}

bool WriteDelimitedTo(const google::protobuf::MessageLite& message,
                      google::protobuf::io::ZeroCopyOutputStream* rawOutput) {
  // We create a new coded stream for each message.  Don't worry, this is fast.
  google::protobuf::io::CodedOutputStream output(rawOutput);

  // Write the size.
  const int size = message.ByteSize();
  output.WriteVarint32(size);

  uint8_t* buffer = output.GetDirectBufferForNBytesAndAdvance(size);
  if (buffer != NULL) {
    // Optimization:  The message fits in one buffer, so use the faster
    // direct-to-array serialization path.
    message.SerializeWithCachedSizesToArray(buffer);
  } else {
    // Slightly-slower path when the message is multiple buffers.
    message.SerializeWithCachedSizes(&output);
    if (output.HadError()) return false;
  }

  return true;
}

std::unique_ptr<google::protobuf::io::ZeroCopyInputStream> OpenForReading(
    const std::string& path) {
  google::protobuf::io::FileInputStream* in =
      new google::protobuf::io::FileInputStream(open(path.c_str(), O_RDONLY));
  in->SetCloseOnDelete(true);
  return std::unique_ptr<google::protobuf::io::ZeroCopyInputStream>(in);
}

bool ReadDelimitedFrom(google::protobuf::io::ZeroCopyInputStream* rawInput,
                       google::protobuf::MessageLite* message) {
  // We create a new coded stream for each message.  Don't worry, this is fast,
  // and it makes sure the 512MB total size limit is imposed per-message rather
  // than on the whole stream.  (See the CodedInputStream interface for more
  // info on this limit.)
  google::protobuf::io::CodedInputStream input(rawInput);
  input.SetTotalBytesLimit(512 * 1024 * 1024, -1);

  // Read the size.
  uint32_t size;
  if (!input.ReadVarint32(&size)) return false;

  // Tell the stream not to read beyond that size.
  google::protobuf::io::CodedInputStream::Limit limit = input.PushLimit(size);

  // Parse the message.
  if (!message->MergeFromCodedStream(&input)) return false;
  if (!input.ConsumedEntireMessage()) return false;

  // Release the limit.
  input.PopLimit(limit);

  return true;
}
