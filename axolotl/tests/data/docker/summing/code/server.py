#!/usr/bin/env python3

import collections
import logging
import pickle
from http import server

import pandas

logger = logging.getLogger(__name__)


class HTTPServer(server.HTTPServer):
    def handle_error(self, request, client_address):
        logger.exception("Exception happened during processing of request from %(client_address)s.", {'client_address': client_address})


class HTTPRequestHandler(server.BaseHTTPRequestHandler):
    def do_POST(self):
        data = self.rfile.read(int(self.headers['Content-Length']))
        # In the future, we should read here just an ObjectId of data
        # in Arrow format in Plasma store and read it from there.
        value = pickle.loads(data)
        sum = self.sum(value)
        result = str(sum).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Length', len(result))
        self.end_headers()
        self.wfile.write(result)

    def sum(self, value):
        if isinstance(value, pandas.DataFrame):
            return sum(self.sum(v) for v in value.itertuples(index=False, name=None))
        if isinstance(value, collections.Iterable):
            return sum(self.sum(v) for v in value)
        else:
            return value

    def log_message(self, message, *args):
        logger.info(message, *args)


if __name__ == '__main__':
    PORT = 8000

    logging.basicConfig(level=logging.INFO)

    logger.info("Listening on port %(port)s.", {'port': PORT})

    httpd = HTTPServer(('', PORT), HTTPRequestHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    logging.info("Server stopped.")
