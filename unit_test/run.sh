#!/usr/bin/env bash
set -e  # Exit on error

# Check dependencies
command -v cmake >/dev/null 2>&1 || { echo "cmake is required but not installed. Aborting." >&2; exit 1; }
command -v make >/dev/null 2>&1 || { echo "make is required but not installed. Aborting." >&2; exit 1; }

# Create data directory if it doesn't exist
mkdir -p data

clear
rm -rf build
mkdir build
cd build

cmake ..
make -j

cd ..

# Run tests
if [ -f ./build/rabitq_unit_test ]; then
    ./build/rabitq_unit_test --gtest_output=xml:./test_result.xml
else
    echo "Error: Test executable not found!"
    exit 1
fi

# Generate coverage report (optional, requires lcov and genhtml)
if command -v lcov >/dev/null 2>&1 && command -v genhtml >/dev/null 2>&1; then
    echo "Generating coverage report..."
    lcov -d . -o test.info -b . -c --rc lcov_branch_coverage=1 2>/dev/null || true
    
    lcov --remove test.info '*12.3.1*' '*gcc*' '/usr/*' '/opt/*' '*/test/*' '*/examples/*' '*/gtest/*' '*/Eigen/*' --output-file coverage_final.info --rc lcov_branch_coverage=1 2>/dev/null || true
    
    genhtml -o ut_output coverage_final.info --rc lcov_branch_coverage=1 2>/dev/null || true
    echo "Coverage report generated in ut_output/"
else
    echo "lcov/genhtml not found. Skipping coverage report generation."
    echo "Install with: sudo apt install lcov"
fi

# lcov -d . -o test.info -b . -c --rc lcov_branch_coverage=1
# lcov --extract test.info '*/fast_scan.h' '*/index_io.h' '*/ivf_rabitQ.h' '*/ivf_rabitQ_search.h' --output-file coverage_final.info --rc lcov_branch_coverage=1
# lcov --remove test.info '*12.3.1*' '*gcc*' '/usr/*' '/opt/*' '*/test/*' '*/examples/*' '*/gtest/*' '*/Eigen/*' --output-file coverage_final.info --rc lcov_branch_coverage=1
# genhtml -o test_output coverage_final.info --rc lcov_branch_coverage=1
