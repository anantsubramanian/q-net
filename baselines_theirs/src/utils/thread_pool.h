#ifndef SRC_UTILS_THREAD_POOL_H_
#define SRC_UTILS_THREAD_POOL_H_

#include <boost/atomic.hpp>
#include <boost/lockfree/queue.hpp>
#include <functional>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  ThreadPool(unsigned int num_threads);
  void Submit(const std::function<void()>& f);
  void ShutDown();

 private:
  void Run();

  std::vector<std::thread> threads_;
  boost::lockfree::queue<std::function<void()>*> queue_;
  boost::atomic<bool> done_;
};

#endif /* SRC_UTILS_THREAD_POOL_H_ */
