# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start from an ubuntu container
FROM ubuntu:focal

# Ensure a SUDO command is available in the container
RUN if type sudo 2>/dev/null; then \
     echo "The sudo command already exists... Skipping."; \
    else \
     echo "#!/bin/sh\n\${@}" > /usr/sbin/sudo; \
     chmod +x /usr/sbin/sudo; \
    fi

# Update & upgrade distribution repositories
RUN apt-get update --fix-missing && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y tzdata --fix-missing; sudo apt upgrade -y --fix-missing

# Install build essentials
RUN sudo apt-get install -y cmake git build-essential;

# Install python pip essentials
RUN sudo apt-get install -y libpython3-dev python3-pip;

# Install VRS dependencies and compile/install VRS
# Note that we install cereal (header only) library to get last version
# On some system libcereal-dev can be enough
RUN sudo apt-get install -y libgtest-dev libgmock-dev \
    libfmt-dev  \
    libturbojpeg-dev libpng-dev \
    liblz4-dev libzstd-dev libxxhash-dev \
    libboost-system-dev libboost-filesystem-dev libboost-thread-dev libboost-chrono-dev libboost-date-time-dev; \
    cd /tmp; git clone https://github.com/USCiLab/cereal.git -b v1.3.2 \
    && cd cereal \
    && cmake -DSKIP_PORTABILITY_TEST=1 -DJUST_INSTALL_CEREAL=ON .; sudo make -j2 install; rm -rf /tmp/cereal;

# Code
ADD ./ /opt/aria_data_tools

# Configure
RUN mkdir /opt/aria_data_tools_Build; cd /opt/aria_data_tools_Build; cmake -DBUILD_WITH_PANGOLIN=OFF /opt/aria_data_tools/src;

# Build & test
RUN cd /opt/aria_data_tools_Build; make -j2 ; ctest -j;

# Build python bindings
RUN cd /opt/aria_data_tools/src; pip3 install --global-option=build_ext --global-option="-j2" .;
