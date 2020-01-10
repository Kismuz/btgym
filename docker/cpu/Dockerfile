FROM debian:stretch

# Install git, supervisor, VNC, & X11 packages
RUN set -ex; \
    apt-get update; \
    apt-get install -y \
      bash \
      fluxbox \
      git \
      net-tools \
      novnc \
      supervisor \
      x11vnc \
      xterm \
      xvfb \
      vim \
      python3 \
      python3-tk \
      python3-pip \
      lsof \
      git \
      libsm6 \
      libxext6 \
      libxrender-dev \
      byobu \
      chromium

RUN python3 -m pip install tornado tensorflow==1.12.0 opencv-python==3.4.4.19 gym==0.10.11 backtrader==1.9.74.123 pyzmq==17.1.2 matplotlib==2.0.2 pillow numpy==1.16.4 scipy==1.3.0 pandas==0.23.4 ipython==7.2.0 psutil==5.4.8 logbook==1.4.1 jupyter jupyter_http_over_ws>=0.0.1a3

# Setup demo environment variables
ENV HOME=/root \
    DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    DISPLAY=:0.0 \
    DISPLAY_WIDTH=1800 \
    DISPLAY_HEIGHT=900 \
    RUN_XTERM=yes \
    RUN_FLUXBOX=yes
COPY . /app
CMD ["/app/entrypoint.sh"]
EXPOSE 8888 8080 6007
