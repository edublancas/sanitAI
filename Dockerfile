FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y \
    supervisor nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /apps

RUN python -m venv /venvs/appenv

COPY requirements.txt /apps/requirements.txt
RUN /venvs/appenv/bin/pip install -r /apps/requirements.txt

RUN /venvs/appenv/bin/python -m spacy download en_core_web_lg

COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisord.conf

COPY . /apps/
RUN rm -f /apps/*.db
RUN /venvs/appenv/bin/pip install -e presidioui/

RUN /venvs/appenv/bin/python -m presidioui.models db
RUN /venvs/appenv/bin/python -m presidioui.models load

EXPOSE 80

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]