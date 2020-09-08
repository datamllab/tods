FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200630-050709

RUN pip3 install -e git+https://gitlab.com/axolotl1/axolotl.git@9619a077e1d06a152fa01f0fca7fa0321dcd3d2c#egg=axolotl
COPY images/Devd3mStart.sh /user_dev/Devd3mStart.sh

RUN chmod a+x /user_dev/Devd3mStart.sh

ENV D3MRUN ta2ta3
ENV TOKENIZERS_PARALLELISM false

EXPOSE 45042

ENTRYPOINT ["/user_dev/Devd3mStart.sh"]
