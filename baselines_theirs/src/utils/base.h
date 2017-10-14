#ifndef SRC_PROTO_BASE_H_
#define SRC_PROTO_BASE_H_

#define CHECK(condition)                                               \
  do {                                                                 \
    if (!(condition)) {                                                \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
              #condition);                                             \
      abort();                                                         \
    }                                                                  \
  } while (0)

#endif /* SRC_PROTO_BASE_H_ */
