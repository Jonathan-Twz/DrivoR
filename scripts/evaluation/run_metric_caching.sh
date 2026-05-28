# ENV variables
export DRIVOR_ROOT="./"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" 
export NUPLAN_MAPS_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset/maps" 
export NAVSIM_EXP_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/DrivoR/exp" 
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DRIVOR_ROOT}" 
export OPENSCENE_DATA_ROOT="/mnt/ws-frb/users/jingyuso/wenzhet/navsim_dataset" 

TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH