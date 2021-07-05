#include <vector>
#include <algorithm>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include "Windows.h"

#define data_size_t size_t
#define label_t size_t


inline int OMP_NUM_THREADS() {
  int ret = 1;
#pragma omp parallel
#pragma omp master
  { ret = omp_get_num_threads(); }
  return ret;
}

template<typename _Iter> inline
static typename std::iterator_traits<_Iter>::value_type* IteratorValType(_Iter) {
  return (0);
}

template<typename _RanIt, typename _Pr> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred) {
  return ParallelSort(_First, _Last, _Pred, IteratorValType(_First));
}


template<typename _RanIt, typename _Pr, typename _VTRanIt> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred, _VTRanIt*) {
  size_t len = _Last - _First;
  const size_t kMinInnerLen = 1024;
  int num_threads = OMP_NUM_THREADS();
  if (len <= kMinInnerLen || num_threads <= 1) {
    std::sort(_First, _Last, _Pred);
    return;
  }
  size_t inner_size = (len + num_threads - 1) / num_threads;
  inner_size = std::max(inner_size, kMinInnerLen);
  num_threads = static_cast<int>((len + inner_size - 1) / inner_size);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < num_threads; ++i) {
    size_t left = inner_size*i;
    size_t right = left + inner_size;
    right = std::min(right, len);
    if (right > left) {
      std::sort(_First + left, _First + right, _Pred);
    }
  }
  // Buffer for merge.
  std::vector<_VTRanIt> temp_buf(len);
  _RanIt buf = temp_buf.begin();
  size_t s = inner_size;
  // Recursive merge
  while (s < len) {
    int loop_size = static_cast<int>((len + s * 2 - 1) / (s * 2));
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < loop_size; ++i) {
      size_t left = i * 2 * s;
      size_t mid = left + s;
      size_t right = mid + s;
      right = std::min(len, right);
      if (mid >= right) { continue; }
      std::copy(_First + left, _First + mid, buf + left);
      std::merge(buf + left, buf + mid, _First + mid, _First + right, _First + left, _Pred);
    }
    s *= 2;
  }
}


double roc_auc(const double* score, const double* label_,  data_size_t num_data_){
  // get indices sorted by score, descent order
  std::vector<data_size_t> sorted_idx;
  for (data_size_t i = 0; i < num_data_; ++i) {
    sorted_idx.emplace_back(i);
  }
  ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
  // temp sum of positive label
  double cur_pos = 0.0f;
  // total sum of positive label
  double sum_pos = 0.0f;
  // accumulate of AUC
  double accum = 0.0f;
  // temp sum of negative label
  double cur_neg = 0.0f;
  double threshold = score[sorted_idx[0]];
  double sum_weights_ = static_cast<double>(num_data_);
  for (data_size_t i = 0; i < num_data_; ++i) {
      const label_t cur_label = label_[sorted_idx[i]];
      const double cur_score = score[sorted_idx[i]];
      // new threshold
      if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
      }
      cur_neg += (cur_label <= 0);
      cur_pos += (cur_label > 0);
  }
  accum += cur_neg*(cur_pos * 0.5f + sum_pos);
  sum_pos += cur_pos;
  double auc = 1.0f;
  if (sum_pos > 0.0f && sum_pos != sum_weights_) {
    auc = accum / (sum_pos *(sum_weights_ - sum_pos));
  }
  return auc;
}


int main(){
  double test_label[500000];
  double test_pred[500000];
  for(int i=0;i<500000;i++){
    test_pred[i] = rand()/RAND_MAX;
    test_label[i] = rand()%2;
  }
  DWORD t1,t2;
  t1 = GetTickCount();
  double auc = roc_auc(test_pred, test_label, 500000);
  t2 = GetTickCount();
  std::cout<< auc << std::endl;
  std::cout<<  (t2 - t1)*1.0/1000 << std::endl;
  system("pause");
  return 0;
}