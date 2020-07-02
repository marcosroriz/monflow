#!/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os

# Função para plotar
color = [0, 0, 255]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Metadado das imagens
# Imagem é dividida em 4 slices
#      ------ 1920 ------              
#      |                |              ---960---
# 1080 |                |  =>  4x  540 | Slice |
#      |                |              --------- 
#      ------------------
# 
# Slices da Imagem
#  -------------------
#  |   0   |    1    |
#  -------------------
#  |   2   |    3    |
#  -------------------
#
# Coordenadas das imagens usa o padrão do opencv
#
# Eixo y: 0 -----------------|
#         |                  |
#         |                  |
#       ymax                 |
# Eixo x: 0---------------xmax

tam_total_width = 1920
tam_total_height = 1080
tam_slice_width = 960
tam_slice_height = 540

# Input, bounding boxes de cada slice
# A entrada é uma lista de boxes
#
# OBS: O campo ID é só para lhe ajudar na hora do merge, no final vou remover)
#
# A entrada abaixo mostra que este slice detectou duas classes (0 e 1)
# Para a classe 0, detectou dois objetos (id 1 e 2) 
# Lembre que o id é só pra te ajudar na hora do merge aqui do teste, na hora não terá
# Ex:
#     [
#            {'id': 1, 'xmin': 1.0, 'ymin': 52.0, 'xmax': 79.0, 'ymax': 230.0, 'conf': 0.93}
#            {'id': 2, 'xmin': 371.0, 'ymin': 3.0, 'xmax': 442.0, 'ymax': 130.0, 'conf': 0.722}
#     ]

# Bounding boxes (entrada)
bb_slice0 = [{'id': 1, 'xmin': 585.0, 'ymin': 396.0, 'xmax': 636.0, 'ymax': 537.0, 'conf': 0.947035014629364},
             {'id': 2, 'xmin': 577.0, 'ymin': 145.0, 'xmax': 610.0, 'ymax': 227.0, 'conf': 0.8426766991615295},
             {'id': 3, 'xmin': 645.0, 'ymin': 122.0, 'xmax': 671.0, 'ymax': 204.0, 'conf': 0.7700527906417847},
             {'id': 4, 'xmin': 669.0, 'ymin': 106.0, 'xmax': 693.0, 'ymax': 191.0, 'conf': 0.635362982749939},
             {'id': 5, 'xmin': 552.0, 'ymin': 487.0, 'xmax': 596.0, 'ymax': 540.0, 'conf': 0.5964713096618652},
             {'id': 6, 'xmin': 707.0, 'ymin': 518.0, 'xmax': 734.0, 'ymax': 539.0, 'conf': 0.3967316150665283}]
bb_slice1 = [{'id': 7, 'xmin': 270.0, 'ymin': 160.0, 'xmax': 314.0, 'ymax': 247.0, 'conf': 0.5448222756385803},
             {'id': 8, 'xmin': 250.0, 'ymin': 146.0, 'xmax': 288.0, 'ymax': 243.0, 'conf': 0.1877407431602478}]
bb_slice2 = [{'id': 9, 'xmin': 647.0, 'ymin': 1.0, 'xmax': 761.0, 'ymax': 169.0, 'conf': 0.9406569600105286},
             {'id': 10, 'xmin': 521.0, 'ymin': 1.0, 'xmax': 605.0, 'ymax': 106.0, 'conf': 0.9205801486968994}]
bb_slice3 = []

# Visualizando os slices
# Img Slice
img_slice0 = os.path.join('data', 'samples', 'test2', '31995_0.jpg')
img_slice1 = os.path.join('data', 'samples', 'test2', '31995_1.jpg')
img_slice2 = os.path.join('data', 'samples', 'test2', '31995_2.jpg')
img_slice3 = os.path.join('data', 'samples', 'test2', '31995_3.jpg')

imgs = [img_slice0, img_slice1, img_slice2, img_slice3]
bbs = [bb_slice0, bb_slice1, bb_slice2, bb_slice3]
for i in range(len(imgs)):
    img = cv2.imread(imgs[i])  # BGR in HxWxC
    for bb in bbs[i]:
        box = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]
        label = 'person' + str(bb['id'])
        plot_one_box(box, img, color, label)

    cv2.imshow("img", img)
    if cv2.waitKey(0) == ord('q'):  # q to quit
        raise StopIteration


# Entrada da função
# Tamanho da iamgem: tam_total_width, tam_total_height, tam_slice_width, tam_slice_height
# bounding boxes de cada slice (usando a ref de cada slice)
#
# Saída:
# uma única lista de bounding boxes, que leva em conta a dimensão da imagem como um todo
# o valor de confiança será o maior dos dois boxes e o quadrante será o maior que junta os dois
#
# Neste caso deve-se fazer o merge do bb de id 5 do slice 0 com o bb de id 10 slice 2
# Também     deve-se fazer o merge do bb de id 6 do slice 0 com o bb de id 9 slice 2
#

def fix_bounding_box(tam_total_width, tam_total_height, tam_slice_width, tam_slice_height,
                     bb_slice0, bb_slice1, bb_slice2, bb_slice3):
    bb_img = []
    return bb_img


bb_img = fix_bounding_box(tam_total_width, tam_total_height, tam_slice_width, tam_slice_height,
                          bb_slice0, bb_slice1, bb_slice2, bb_slice3)

# Imagem Total
img_total = os.path.join('data', 'samples', 'test2', '31930.jpg')
img = cv2.imread(img_total)  # BGR in HxWxC

for bb in bb_img:
    box = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]
    label = 'person' + str(bb['id'])
    plot_one_box(box, img, color, label)

cv2.imshow("img", img)
if cv2.waitKey(0) == ord('q'):  # q to quit
    raise StopIteration
