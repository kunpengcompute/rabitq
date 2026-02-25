/*
   Copyright 2026 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
#include "test_result.h"
#include <algorithm>

TestResult::TestResult()
    : total_time(0), recall(0), calculate_quantity(0), build_time(0) {
    search_time.clear();
    accumulate_time.clear();
    }

TestResult::~TestResult() {
    search_time.clear();
    accumulate_time.clear();
}

void TestResult::reorder() {
    if (search_time.size() > 0) {
        std::sort(search_time.begin(), search_time.end());
        accumulate_time.clear();
        accumulate_time.push_back(0);
        for (size_t i = 0; i < search_time.size(); ++i) {
            accumulate_time.push_back(accumulate_time.back() + search_time[i]);
        }
    }
}

double TestResult::get_total(int begin, int end) {
    if (begin >= 0 && end < accumulate_time.size() && begin < end) {
        return accumulate_time[end] - accumulate_time[begin];
    }
    return 0;
}