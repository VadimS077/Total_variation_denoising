import cv2
import numpy as np


imgn = cv2.imread("C:/users/vadim/PycharmProjects/pythonProject/test.jpg",cv2.IMREAD_GRAYSCALE)
#Добавление шума
def add_noise(image,mean=0, stddev=60):
    row, col = image.shape
    noise = np.random.normal(mean, stddev, (row, col))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image
img_obs = add_noise(imgn,0,30)
cv2.imshow("old",img_obs)
#total_variation_denoising
def TVLdenoise(im, lmbda, niter=100):
    L2 = 8
    tau = 0.02
    sigma = 1.0 / (L2 * tau)
    theta = 1
    lt = lmbda * tau
    height, width = im.shape[:2]
    unew = np.zeros((height, width))
    p = np.zeros((height, width, 2))
    d = np.zeros((height, width))
    ux = np.zeros((height, width))
    uy = np.zeros((height, width))
    mx = np.max(im)
    if mx > 1.0:
        nim = im.astype(np.float64) / mx  # normalize
    else:
        nim = im.astype(np.float64)  # leave intact
    u = nim.copy()
    p[:, :, 0] = u[:, np.r_[1:width, width - 1]] - u


    p[:, :, 1] = u[np.r_[1:height, height - 1], :] - u

    for k in range(niter):
        ux = (u[:, np.r_[1:width, width - 1]] - u)*sigma
        uy = (u[np.r_[1:height, height - 1], :] - u)*sigma

        p = p + np.concatenate((ux[..., np.newaxis], uy[..., np.newaxis]), axis=2)
        normep = np.maximum(1, np.sqrt(p[:, :, 0] ** 2 + p[:, :, 1] ** 2))


        p[:, :, 0] = p[:, :, 0] / normep
        p[:, :, 1] = p[:, :, 1] / normep
        div = np.vstack((p[:-1, :, 1], np.zeros((1, width)))) - np.vstack((np.zeros((1, width)), p[:-1, :, 1]))

        div += np.hstack((p[:, :-1, 0], np.zeros((height, 1)))) - np.hstack((np.zeros((height, 1)), p[:, :-1, 0]))
        #another form for L2
        #unew = (u + tau * div + lt * nim) / (1 + tau)

        v = u+tau*div
        unew = (v - lt) * ((v - nim) > lt) + (v + lt) * ((v - nim) < -lt) + nim * (np.abs(v - nim) <= lt)
        u = unew+theta*(unew-u)

    return u


imgg = TVLdenoise(img_obs, 2, niter=50)
cv2.imshow("new1", imgg)
cv2.waitKey(0)





