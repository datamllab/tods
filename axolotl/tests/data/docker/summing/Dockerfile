FROM registry.gitlab.com/datadrivendiscovery/images/base:ubuntu-bionic-python36

EXPOSE 8000/tcp

RUN apt-get update -q -q && \
 apt-get install --yes --force-yes runit

COPY ./runsvdir-start /usr/local/sbin/runsvdir-start
COPY ./requirements.txt /requirements.txt
COPY ./code /code
COPY ./etc /etc

RUN pip3 install -r /requirements.txt && rm -f /requirements.txt

WORKDIR /code

ENTRYPOINT ["/usr/local/sbin/runsvdir-start"]
