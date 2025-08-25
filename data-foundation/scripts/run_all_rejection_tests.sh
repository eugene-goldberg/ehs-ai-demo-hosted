#!/bin/bash
# Run all rejection tests with logging

TEST_DIR="/tmp/rejection_test_logs"
mkdir -p $TEST_DIR

# Run each test in background
python -m pytest test/test_document_recognition_service.py -v > $TEST_DIR/recognition.log 2>&1 &
python -m pytest test/test_rejection_decision_engine.py -v > $TEST_DIR/decision.log 2>&1 &
python -m pytest test/test_rejected_document_service.py -v > $TEST_DIR/storage.log 2>&1 &
python -m pytest test/test_rejection_tracking_api.py -v > $TEST_DIR/api.log 2>&1 &
python -m pytest test/test_ingestion_workflow_with_rejection.py -v > $TEST_DIR/workflow.log 2>&1 &
python -m pytest test/test_rejection_integration_e2e.py -v > $TEST_DIR/e2e.log 2>&1 &

# Monitor all logs
tail -f $TEST_DIR/*.log
EOF

chmod +x run_all_rejection_tests.sh
