import numpy as np

def convolution2d(image, kernel, bias):
    m, n = kernel.shape

    # Добовляем подушку с шириной sh
    sh = 1
    image = np.pad(image, sh, 'reflect')

    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

def mozaik(R,G,B):

    # Создаём копии, так как аргументы это ссылки
    R = R.copy()
    G = G.copy()
    B = B.copy()

    # Стираем информацию о цвете в нужных местах.
    G[::2, ::2] = 0
    G[1::2, 1::2] = 0

    B[::2, 1::2] = 0
    B[1::2, :] = 0

    R[::2, :] = 0
    R[1::2, 1::2] = 0

    return R,G,B

# def restore_chanels_RGB(R,G,B):
#
#     # Создаём копии, так как аргументы это ссылки
#     R = R.copy()
#     G = G.copy()
#     B = B.copy()
#
#     # R = np.arange(100)
#     # R = R[::-1]
#     # R = np.reshape(R, (10, 10))
#
#     # Добовляем подушку с шириной sh
#     sh = 2
#     # Лучше сделать симетричное отрожение а не конст валью
#     R = np.pad(R, sh, 'constant', constant_values=0)
#     G = np.pad(G, sh, 'constant', constant_values =0 )
#     B = np.pad(B, sh, 'constant', constant_values=0)
#
#
#     # Востоновление зелёного канала
#     for i in range(sh, G.shape[0] - sh, 1):
#         for j in range(sh, G.shape[1] - sh, 2):
#             j = j + (i % 2)
#             G[i][j] = (np.longlong(G[i][j+1]) + G[i][j - 1] + G[i + 1][j] + G[i - 1][j]) / 4
#
#     # Востоновление Красного канала
#     for i in range(sh, G.shape[0] - sh, 2):
#         for j in range(sh + 1, G.shape[1] - sh, 2):
#             R[i][j] = (np.longlong(R[i + 1][j + 1]) + R[i + 1][j - 1] + R[i - 1][j + 1] + R[i - 1][j - 1]) / 4
#
#     # Востоновление Синиго канала
#     for i in range(sh + 1, G.shape[0] - sh, 2):
#         for j in range(sh + 1, G.shape[1] - sh, 2):
#             B[i][j] = (np.longlong(B[i + 1][j + 1]) + B[i + 1][j - 1] + B[i - 1][j + 1] + B[i - 1][j - 1]) / 4
#
#     # Убираем подушку с шириной sh
#     R = R[sh:-sh, sh:-sh]
#     G = G[sh:-sh, sh:-sh]
#     B = B[sh:-sh, sh:-sh]
#
#     return R,G,B



