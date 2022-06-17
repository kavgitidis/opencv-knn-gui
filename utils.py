import io
import os

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from pymongo import MongoClient
from pymongo.errors import DocumentTooLarge
from scipy.spatial import distance


class MediaDB:
    def __init__(self, client: str, database: str, collection: str):
        # cluster = MongoClient('mongodb://localhost:27017/')
        # db = cluster["adb"]
        # collection = db["adb"]
        self.cluster = MongoClient(client)
        self.db = self.cluster[database]
        self.collection = self.db[collection]

    def put_images_to_db(self, folder_path: str):
        """

        Args:
            folder_path:

        Returns:

        """
        files = os.listdir(folder_path)
        i = ImageUtils()
        count = 0
        for file in files:
            full_path = folder_path + "/" + file
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            img = imutils.resize(img, width=475)
            himg = i.read_image(full_path)
            himg = imutils.resize(himg, width=475)
            a, b, (keypoints_sift, descriptors_sift) = i.sift(img)
            a, b, (keypoints_surf, descriptors_surf) = i.surf(img)
            a, b, (keypoints_orb, descriptors_orb) = i.orb(img)
            descriptors_hist = i.histogram(himg)

            post = {"ImageName": file, "ImagePath": f"{full_path}",
                    "SIFTDescriptors": descriptors_sift.tolist(),
                    "SIFTKeypoints": i.encode_keypoints(keypoints_sift),
                    "SURFDescriptors": descriptors_surf.tolist(),
                    "SURFKeypoints": i.encode_keypoints(keypoints_surf),
                    "ORBDescriptors": descriptors_orb.tolist(), "ORBKeypoints": i.encode_keypoints(keypoints_orb),
                    "HISTOGRAMDescriptors": descriptors_hist.tolist(), }
            try:
                self.collection.insert_one(post)
                print(f"Inserted {file} in MongoDB.")
                count += 1
            except DocumentTooLarge:
                print(f"{file} was not inserted in MongoDB.")

        return count


class ImageUtils:
    def __init__(self):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._surf = cv2.xfeatures2d.SURF_create()
        self._orb = cv2.ORB_create(nfeatures=1500)

    def read_image(self, path, color=cv2.IMREAD_COLOR):
        """

        Args:
            color:
            path:

        Returns:

        """
        img = cv2.imread(path, color)
        return imutils.resize(img, width=475)

    def sift(self, img):
        """

        Args:
            img:

        Returns:

        """
        index_params = dict(algorithm=0, trees=5)
        return np.float32, index_params, self._sift.detectAndCompute(img, None)

    def surf(self, img):
        """

        Args:
            img:

        Returns:

        """
        index_params = dict(algorithm=0, trees=5)
        return np.float32, index_params, self._surf.detectAndCompute(img, None)

    def orb(self, img):
        """

        Args:
            img:

        Returns:

        """
        index_params = dict(algorithm=6,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        return np.uint8, index_params, self._orb.detectAndCompute(img, None)

    def histogram(self, img):
        """

        Args:
            img:

        Returns:

        """
        hist = cv2.calcHist([img],
                            [0, 1, 2],
                            None,
                            [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def get_knn(self, query_img: str, method: str, algorithm: str, k: int, db: MediaDB, dist: str, ratio: float,
                dist_metric=cv2.NORM_L2):
        """

        Args:
            dist_metric:
            ratio:
            dist:
            query_img:
            method:
            algorithm:
            k:
            db:

        Returns:

        """
        knns = []
        if algorithm == 'SURF' or algorithm == 'SIFT' or algorithm == 'ORB':
            if method == 'FLANN':
                dtype_, index_params, (keypoints, descriptors) = getattr(self, algorithm.lower())(
                    self.read_image(query_img, cv2.IMREAD_GRAYSCALE))

                for img in db.collection.find({'ImagePath': {'$ne': f'{query_img}'}}):
                    good_matches = 0

                    des = np.asarray(img[f'{algorithm}Descriptors'], dtype=dtype_)

                    search_params = dict(checks=50)  # or pass empty dictionary

                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(descriptors, des, k=2)
                    for i, pair in enumerate(matches):
                        try:
                            m, n = pair
                            if m.distance < ratio * n.distance:
                                good_matches += 1
                        except ValueError:
                            pass
                    precision = (good_matches / len(matches)) * 100
                    knns.append({'ImageName': img['ImageName'], 'ImagePath': img['ImagePath'],
                                 'Keypoints': img[f'{algorithm}Keypoints'], "Matches": matches,
                                 "Precision": precision,
                                 "Count": good_matches})
                sorted_ = sorted(knns, key=lambda x: x['Count'], reverse=True)[:k]
                average_precision = sum([i['Precision'] for i in sorted_]) / len(sorted_)
                return average_precision, sorted(knns, key=lambda x: x['Count'], reverse=True)[:k]
            elif method == 'BF':
                if algorithm == 'SURF' or algorithm == 'SIFT' or algorithm == 'ORB':
                    dtype_, index_params, (keypoints, descriptors) = getattr(self, algorithm.lower())(
                        self.read_image(query_img, cv2.IMREAD_GRAYSCALE))
                    dist_metric = getattr(cv2, str(dist_metric))
                    for img in db.collection.find({'ImagePath': {'$ne': f'{query_img}'}}):
                        good_matches = 0

                        des = np.asarray(img[f'{algorithm}Descriptors'], dtype=dtype_)
                        bf = cv2.BFMatcher(dist_metric)  # crossCheck=True
                        matches = bf.knnMatch(descriptors, des, k=2)
                        for i, pair in enumerate(matches):
                            try:
                                m, n = pair
                                if m.distance < ratio * n.distance:
                                    good_matches += 1
                            except ValueError:
                                pass
                        precision = (good_matches / len(matches)) * 100
                        knns.append({'ImageName': img['ImageName'], 'ImagePath': img['ImagePath'],
                                     'Keypoints': img[f'{algorithm}Keypoints'], "Matches": matches,
                                     "Precision": precision,
                                     "Count": good_matches})
                    sorted_ = sorted(knns, key=lambda x: x['Precision'], reverse=True)[:k]
                    average_precision = sum([i['Precision'] for i in sorted_]) / len(sorted_)
                    return average_precision, sorted_
        elif algorithm == 'HISTOGRAM':
            for img in db.collection.find(({'ImagePath': {'$ne': f'{query_img}'}})):
                hist1 = self.histogram(self.read_image(query_img))
                hist2 = np.asarray(img[f'{algorithm}Descriptors'], dtype=np.float32)
                d = abs(getattr(distance, dist)(hist1, hist2))
                knns.append(
                    {'ImageName': img['ImageName'], 'ImagePath': img['ImagePath'], "Distance": d,
                     "Precision": (1 - d) * 100})
            sorted_ = sorted(knns, key=lambda x: x['Precision'], reverse=True)[:k]
            average_precision = sum([i['Precision'] for i in sorted_]) / len(sorted_)
            return average_precision, sorted_

    def encode_keypoints(self, keypoints):
        """

        Args:
            keypoints:

        Returns:

        """
        return [[keypoint.angle,
                 keypoint.class_id,
                 keypoint.octave,
                 keypoint.pt,
                 keypoint.response,
                 keypoint.size] for keypoint in keypoints]

    def decode_keypoints(self, keypoints):
        """

        Args:
            keypoints:

        Returns:

        """
        return [cv2.KeyPoint(_angle=keypoint[0],
                             _class_id=keypoint[1],
                             _octave=keypoint[2],
                             x=keypoint[3][0],
                             y=keypoint[3][1],
                             _response=keypoint[4],
                             _size=keypoint[5]) for keypoint in keypoints]

    def draw_matches(self, img1=None, img2=None, kp2=None, algorithm=None, matches=None, ratio=None):
        """

        Args:
            img1:
            img2:
            kp2:
            algorithm:
            matches:
            ratio:

        Returns:

        """
        if algorithm == 'HISTOGRAM':
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            img = cv2.imread(img2)

            colors = ('b', 'g', 'r')
            chans = cv2.split(img)
            chans = chans
            for (chan, color) in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [254], [1, 255])
                hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                ax1.plot(hist, color=color)
            image = plt.imread(img2)
            ax1.imshow(image, extent=[1, 255, 1, 255])
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=255)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img
        else:
            img1 = self.read_image(img1, cv2.IMREAD_GRAYSCALE)
            img2 = self.read_image(img2, cv2.IMREAD_GRAYSCALE)
            dtype_, index_params, (kp1, descriptors) = getattr(self, algorithm.lower())(
                img1)
            matches_mask = [[0, 0] for _ in range(len(matches))]
            for i, pair in enumerate(matches):
                try:
                    m, n = pair
                    if m.distance < ratio * n.distance:
                        matches_mask[i] = [1, 0]
                except ValueError:
                    pass
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matches_mask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        return img3
