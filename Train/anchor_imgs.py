from generate_anchors import generate_anchors
import numpy as np
import cv2
import random

class VideoBuilder():
    """
    Produces images respect to video, enhance if possible
    """
    def __init__(self, ):
        anchor_class = AnchorTarget()
        self._anchors = anchor_class.forward_anchors().astype(np.int)

        '''
        self._image_dir = image_dir
        self._video_list = video_list
        self._n_frames = n_frames
        '''
    def _cropSingleImage(self, image):
        image_anchors = list()
        for _anchor in self._anchors:
            x, y, X, Y = _anchor
            image_anchors.append(image[y:Y, x:X, :])
        return image_anchors

    #crop Images using anchors
    def _anchorImages(self, image_list):
        image_list_anchors = list()
        for raw_image in image_list:
            image_anchors = self._cropSingleImage(raw_image)
            image_list_anchors += image_anchors
        return image_list_anchors

    def video2Pictures(self, video_paths, image_dir, num_frames=200):
        image_list = list()
        for video_path in video_paths:
            try:
                cap = cv2.VideoCapture(video_path)
            except Exception as e:
                print('video loading error')

            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 1)
                image_list.append(frame)
            #Release everything if job is finished
            cap.release()
        #shuffle video frames
        random.shuffle(image_list)
        #preprocess images or enhance images
        sv_images_list = self._anchorImages(image_list[:num_frames])
        for idx, sv_image in enumerate(sv_images_list):
            cv2.imwrite(image_dir + '/test_'+ str(idx) + '.jpg', sv_image) 
        return 1



class AnchorTarget():
    """
    Produces anchors with different scales and swifts
    """
    def __init__(self, anchor_scales=(1,), stride=80, bz_size=400):
        self._base_size = bz_size
        self._anchors = generate_anchors(base_size=self._base_size, ratios=[1,], scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = stride 

    def forward_anchors(self, height=640, width=480):
        x_size = int((width-self._base_size) / self._feat_stride) + 1
        y_size = int((height-self._base_size) / self._feat_stride) + 1
        #shift_x = np.arange(0, x_size) * self._feat_stride
        #shift_y = np.arange(0, y_size) * self._feat_stride
        shift_x = np.arange(0, x_size) * self._feat_stride
        shift_y = np.array([0, 3]) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), 
                shift_x.ravel(), shift_y.ravel())).transpose()

        center_shift = np.array([(width - self._base_size)/2, (height - self._base_size)/2, (width - self._base_size)/2, (height - self._base_size)/2]) 
        shifts = np.vstack( (shifts, center_shift) )

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)
        return all_anchors


if __name__ == '__main__':
    '''
    anchor_class = AnchorTarget()
    anchors = anchor_class.forward_anchors()
    '''
    vbuilder =  VideoBuilder()
    vbuilder.video2Pictures(['/mnt/exhdd/tomorning_dataset/wonderland/cv/deep_retriver/imgs_retriver/Oss_task_2/26500/26500_0.mov'], '/mnt/exhdd/tomorning_dataset/wonderland/cv/deep_retriver/pyTrainVideo/test_imgs', 20)
