#!/bin/bash
set -ex

RUN_FLUXBOX=${RUN_FLUXBOX:-yes}
RUN_XTERM=${RUN_XTERM:-yes}

case $RUN_FLUXBOX in
  false|no|n|0)
    rm -f /app/conf.d/fluxbox.conf
    ;;
esac

case $RUN_XTERM in
  false|no|n|0)
    rm -f /app/conf.d/xterm.conf
    ;;
esac
exec git clone https://github.com/Kismuz/btgym.git /workspace &
exec x11vnc -storepasswd "btgym" /app/x11vnc.pass && chmod a+r /app/x11vnc.pass &
exec jupyter serverextension enable --py jupyter_http_over_ws &
exec tensorboard --logdir=/workspace/btgym/logdir &
exec jupyter notebook --notebook-dir=/workspace/ --ip 0.0.0.0 --allow-root --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0 &
exec supervisord -c /app/supervisord.conf
