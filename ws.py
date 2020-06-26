#!/bin/env python
# -*- coding: utf-8 -*-

"""MonFlow Web Service Interface"""
import base64
import cv2
import click
import logging
import numpy as np

from spyne import Application, rpc, ServiceBase, ComplexModel, \
    String, Integer, DateTime, File, ByteArray, Mandatory, Decimal

from spyne.protocol.http import HttpRpc
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication


class ImageRequest(ComplexModel):
    imgID = Integer
    cameraID = Integer
    timestamp = DateTime
    file = File


class ImageReply(ComplexModel):
    imgID = Integer
    qtdPessoas = Integer
    qtdVeiculos = Integer
    qtdCarro = Integer
    qtdMoto = Integer
    qtdCaminhao = Integer
    qtdOnibus = Integer


class WSAPI(ServiceBase):
    @rpc(ImageRequest, _returns=ImageReply)
    def process_img(ctx, request):
        print("IMG ID", request.imgID)
        dadosImg = request.file.data[0]
        print(type(dadosImg))
        print(len(dadosImg))
        print(dadosImg[:100])

        # nparr = np.fromstring(base64.b64decode(dadosImg), np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        return ImageReply(imgID=request.imgID, qtdPessoas=5, qtdVeiculos=3,
                          qtdCarro=2, qtdMoto=1, qtdCaminhao=0, qtdOnibus=0)


application = Application([WSAPI],
                          tns='monflow',
                          in_protocol=Soap11(validator='lxml'),
                          out_protocol=Soap11(validator='lxml')
                          )

if __name__ == '__main__':
    # You can use any Wsgi server. Here, we chose
    # Python's built-in wsgi server but you're not
    # supposed to use it in production.
    from wsgiref.simple_server import make_server
    wsgi_app = WsgiApplication(application)
    server = make_server('0.0.0.0', 9000, wsgi_app)
    server.serve_forever()
