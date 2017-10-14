#include "utils/thread_pool.h"

#include <iostream>

using namespace std;

ThreadPool::ThreadPool(unsigned int num_threads) : queue_(0), done_(false) {
  for (int i = 0; i < num_threads; ++i) {
    threads_.emplace_back(&ThreadPool::Run, this);
  }
}

void ThreadPool::Submit(const std::function<void()>& f) {
  queue_.push(new std::function<void()>(f));
}

void ThreadPool::ShutDown() {
  done_ = true;
  for (thread& thread : threads_) {
    thread.join();
  }
}

void ThreadPool::Run() {
  std::function<void()>* f;
  while (!done_) {
    while (queue_.pop(f)) {
      (*f)();
      delete f;
    }
  }
  while (queue_.pop(f)) {
    (*f)();
    delete f;
  }
}
