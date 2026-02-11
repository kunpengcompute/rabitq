#!/usr/bin/env bash
clear
rm -rf build
mkdir build
cd build

cmake ..
make -j

cd ..

./build/rabitq_unit_test --gtest_output=xml:./test_result.xml

lcov -d . -o test.info -b . -c --rc lcov_branch_coverage=1

lcov --remove test.info '*12.3.1*' '*gcc*' '/usr/*' '/opt/*' '*/test/*' '*/examples/*' '*/gtest/*' '*/Eigen/*' --output-file coverage_final.info --rc lcov_branch_coverage=1

genhtml -o ut_output coverage_final.info --rc lcov_branch_coverage=1

# lcov -d . -o test.info -b . -c --rc lcov_branch_coverage=1
# lcov --extract test.info '*/fast_scan.h' '*/index_io.h' '*/ivf_rabitQ.h' '*/ivf_rabitQ_search.h' --output-file coverage_final.info --rc lcov_branch_coverage=1
# lcov --remove test.info '*12.3.1*' '*gcc*' '/usr/*' '/opt/*' '*/test/*' '*/examples/*' '*/gtest/*' '*/Eigen/*' --output-file coverage_final.info --rc lcov_branch_coverage=1
# genhtml -o test_output coverage_final.info --rc lcov_branch_coverage=1
