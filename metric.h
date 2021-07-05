#ifndef METRIC_HPP
#define METRIC_HPP

 
#ifdef __cplusplus
extern "C" {
#endif

#define data_size_t size_t
#define label_t size_t

__declspec(dllexport) double __cdecl c_roc_auc_score(const double* score, const double* label_,  data_size_t num_data_);

#ifdef __cplusplus
}
#endif

#endif