NAME=pyds_tracker_meta
PKGS="gstreamer-1.0 gstreamer-video-1.0"
NVDS_VERSION="5.0"
NVDS_PATH="/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/sources/includes/"

c++ -O3 -Wall -shared -std=c++11 \
    -fPIC `python3 -m pybind11 --includes` \
    pyds_tracker_meta.cpp \
    -o ${NAME}`python3-config --extension-suffix` \
    -I${NVDS_PATH} \
    `pkg-config --cflags ${PKGS}`
