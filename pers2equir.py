"""
Code adapted from https://github.com/timy90022/
"""
import cv2
import numpy as np


class Perspective:
    def __init__(self, img, FOV, THETA, PHI):
        if isinstance(img, str):
            self._img = cv2.imread(img, cv2.IMREAD_COLOR)
        else:
            self._img = img
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    def GetEquirec(self, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        x, y = np.meshgrid(np.linspace(-180, 180, width, dtype=np.float32), np.linspace(90, -90, height, dtype=np.float32))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map), axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height, width, 3])
        inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

        xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

        lon_map = np.where((-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) & (-self.h_len < xyz[:, :, 2])
                           & (xyz[:, :, 2] < self.h_len), (xyz[:, :, 1] + self.w_len) / 2 / self.w_len * self._width, 0)
        lat_map = np.where((-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) & (-self.h_len < xyz[:, :, 2])
                           & (xyz[:, :, 2] < self.h_len), (-xyz[:, :, 2] + self.h_len) / 2 / self.h_len * self._height,
                           0)
        mask = np.where((-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) & (-self.h_len < xyz[:, :, 2])
                        & (xyz[:, :, 2] < self.h_len), 1, 0)

        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
                          # borderMode=cv2.BORDER_WRAP)

        channels = self._img.shape[-1]
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)
        persp = persp * mask

        return persp, mask


class MultiPerspective:
    def __init__(self, img_array, F_T_P_array):  # list of (fov, azimuth deg, elevation deg)
        assert len(img_array) == len(F_T_P_array)
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

    def GetEquirec(self, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height,width,3), dtype=np.float32)
        merge_mask = np.zeros((height,width,3), dtype=np.float32)

        for img,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            per = Perspective(img,F,T,P)        # Load equirectangular image
            img, mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)

            merge_image += img
            merge_mask += mask

        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))

        return merge_image


class MultiPerspectiveWeighted:
    def __init__(self, img_array, weights, F_T_P_array):  # list of (fov, azimuth deg, elevation deg)
        assert len(img_array) == len(F_T_P_array)
        self.img_array = img_array
        self.weights = weights
        self.F_T_P_array = F_T_P_array

    def GetEquirec(self, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, 3), dtype=np.float32)
        total_weight = np.zeros((height, width, 3), dtype=np.float32)

        for img, weight, [F, T, P] in zip (self.img_array, self.weights, self.F_T_P_array):
            imgw = np.concatenate((img, weight[..., None]), axis=-1)
            per = Perspective(imgw, F, T, P)        # Load equirectangular image
            imgw, mask = per.GetEquirec(height, width)   # Specify parameters(FOV, theta, phi, height, width)

            img = imgw[..., :3]
            w = imgw[..., 3:]

            merge_image += img * w
            total_weight += w

        total_weight = np.where(total_weight == 0, 1, total_weight)  # avoid dividing by 0
        merge_image = merge_image / total_weight

        return merge_image
