@echo off
REM =============================================================================
REM Environment variables configuration to suppress warnings
REM =============================================================================

REM Configure warning filters for Python
set PYTHONWARNINGS=ignore::UserWarning:lightning_utilities,ignore::FutureWarning:sklearn,ignore::DeprecationWarning:pkg_resources

REM Configure TensorFlow logging
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

echo Environment variables configured to suppress non-critical warnings
